"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vliw(self, slots: list[tuple[Engine, tuple]]) -> dict:
        """Pack slots into a single VLIW instruction bundle."""
        instr = defaultdict(list)
        for engine, slot in slots:
            instr[engine].append(slot)
        return dict(instr)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW+SIMD implementation.
        Process VLEN (8) elements at once using vector operations.
        Pack independent operations into same instruction bundle.
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Initialize memory layout variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Vector temporaries (VLEN = 8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Broadcast constants to vectors
        self.instrs.append(self.build_vliw([
            ("valu", ("vbroadcast", v_zero, zero_const)),
            ("valu", ("vbroadcast", v_one, one_const)),
            ("valu", ("vbroadcast", v_two, two_const)),
            ("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])),
        ]))

        # Precompute hash stage constants as vectors
        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.alloc_scratch(f"v_hash_{hi}_c1", VLEN)
            v_const3 = self.alloc_scratch(f"v_hash_{hi}_c3", VLEN)
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            self.instrs.append(self.build_vliw([
                ("valu", ("vbroadcast", v_const1, c1_scalar)),
                ("valu", ("vbroadcast", v_const3, c3_scalar)),
            ]))
            v_hash_consts.append((v_const1, v_const3))

        # Process groups with pipelining across batches
        n_groups_total = batch_size // VLEN  # 32
        BATCH_SIZE = 8  # Process 8 groups at a time (scratch-constrained)
        n_batches = n_groups_total // BATCH_SIZE  # 4 batches per round

        # Shared temporaries for valu (only one valu active at a time)
        shared_tmp1 = [self.alloc_scratch(f"shared_tmp1_{g}", VLEN) for g in range(BATCH_SIZE)]
        shared_tmp2 = [self.alloc_scratch(f"shared_tmp2_{g}", VLEN) for g in range(BATCH_SIZE)]

        # Allocate FOUR buffers - eliminates aliasing conflicts!
        # With 4 buffers for 4 batches: no intra-round conflicts
        # With round-offset indexing: no cross-round conflicts either
        def alloc_batch_scratch(prefix):
            return {
                'addr_idx': [self.alloc_scratch(f"{prefix}_{g}_addr_idx") for g in range(BATCH_SIZE)],
                'addr_val': [self.alloc_scratch(f"{prefix}_{g}_addr_val") for g in range(BATCH_SIZE)],
                'ga': [[self.alloc_scratch(f"{prefix}_{g}_ga_{i}") for i in range(VLEN)] for g in range(BATCH_SIZE)],
                'idx': [self.alloc_scratch(f"{prefix}_{g}_idx", VLEN) for g in range(BATCH_SIZE)],
                'val': [self.alloc_scratch(f"{prefix}_{g}_val", VLEN) for g in range(BATCH_SIZE)],
                'node': [self.alloc_scratch(f"{prefix}_{g}_node", VLEN) for g in range(BATCH_SIZE)],
                'tmp1': shared_tmp1,  # Shared!
                'tmp2': shared_tmp2,  # Shared!
            }

        # 4 buffers: no aliasing within a round, enables deeper pipelining
        buf = [alloc_batch_scratch("A"), alloc_batch_scratch("B"),
               alloc_batch_scratch("C"), alloc_batch_scratch("D")]

        def build_valu_ops(b):
            """Build VALU ops for batch b with modulo-scheduled hash for better pipelining"""
            ops_list = []

            # MODULO-SCHEDULED HASH with XOR overlap
            def tmp_ops(hi, groups):
                """Generate tmp1, tmp2 ops for given hash stage and groups"""
                v_c1, v_c3 = v_hash_consts[hi]
                ops = []
                for g in groups:
                    ops.append(("valu", (HASH_STAGES[hi][0], b['tmp1'][g], b['val'][g], v_c1)))
                    ops.append(("valu", (HASH_STAGES[hi][3], b['tmp2'][g], b['val'][g], v_c3)))
                return ops

            def combine_ops(hi, groups):
                """Generate combine ops for given hash stage and groups"""
                return [("valu", (HASH_STAGES[hi][2], b['val'][g], b['tmp1'][g], b['tmp2'][g]))
                        for g in groups]

            # XOR for g0-5, then XOR g6-7 + hash stage 0 tmp g0-1
            ops_list.append([("valu", ("^", b['val'][g], b['val'][g], b['node'][g])) for g in range(6)])
            xor_67 = [("valu", ("^", b['val'][g], b['val'][g], b['node'][g])) for g in [6, 7]]
            ops_list.append(xor_67 + tmp_ops(0, [0, 1]))  # 2 + 4 = 6 ops

            # Stage 0 tmp for remaining groups (2 cycles)
            ops_list.append(tmp_ops(0, [2, 3, 4]))  # 6 ops
            ops_list.append(tmp_ops(0, [5, 6, 7]))  # 6 ops

            # Pipelined stages 1-5: overlap combine[hi-1] with tmp[hi]
            for hi in range(1, 6):
                # g0-5 combine from prev stage (6 ops)
                ops_list.append(combine_ops(hi-1, [0, 1, 2, 3, 4, 5]))
                # g6-7 combine prev (2 ops) + g0-1 tmp cur (4 ops) = 6 ops
                ops_list.append(combine_ops(hi-1, [6, 7]) + tmp_ops(hi, [0, 1]))
                # g2-4 tmp cur (6 ops)
                ops_list.append(tmp_ops(hi, [2, 3, 4]))
                # g5-7 tmp cur (6 ops)
                ops_list.append(tmp_ops(hi, [5, 6, 7]))

            # Epilogue + Index: Aggressive overlap of under-utilized cycles
            # Cycle 1: g0-5 combine stage 5 (6 ops)
            ops_list.append(combine_ops(5, [0, 1, 2, 3, 4, 5]))
            # Cycle 2: g6-7 combine (2) + g0-1 & + multiply_add (4) = 6 ops
            idx_ops_01 = []
            for g in [0, 1]:
                idx_ops_01.append(("valu", ("&", b['tmp1'][g], b['val'][g], v_one)))
                idx_ops_01.append(("valu", ("multiply_add", b['idx'][g], b['idx'][g], v_two, v_one)))
            ops_list.append(combine_ops(5, [6, 7]) + idx_ops_01)
            # Cycle 3: g2-4 & + multiply_add (6 ops)
            idx_ops_24 = []
            for g in [2, 3, 4]:
                idx_ops_24.append(("valu", ("&", b['tmp1'][g], b['val'][g], v_one)))
                idx_ops_24.append(("valu", ("multiply_add", b['idx'][g], b['idx'][g], v_two, v_one)))
            ops_list.append(idx_ops_24)
            # Cycle 4: g5-7 & + multiply_add (6 ops)
            idx_ops_57 = []
            for g in [5, 6, 7]:
                idx_ops_57.append(("valu", ("&", b['tmp1'][g], b['val'][g], v_one)))
                idx_ops_57.append(("valu", ("multiply_add", b['idx'][g], b['idx'][g], v_two, v_one)))
            ops_list.append(idx_ops_57)
            # Cycle 5: add g0-5 (6 ops)
            ops_list.append([("valu", ("+", b['idx'][g], b['idx'][g], b['tmp1'][g])) for g in range(6)])
            # Cycle 6: add g6-7 (2) + bounds g0-3 (4) = 6 ops
            add_67 = [("valu", ("+", b['idx'][g], b['idx'][g], b['tmp1'][g])) for g in [6, 7]]
            bounds_03 = [("valu", ("<", b['tmp1'][g], b['idx'][g], v_n_nodes)) for g in range(4)]
            ops_list.append(add_67 + bounds_03)
            # Cycle 7: bounds g4-7 (4) + multiply_mask g0-1 (2) = 6 ops
            bounds_47 = [("valu", ("<", b['tmp1'][g], b['idx'][g], v_n_nodes)) for g in [4, 5, 6, 7]]
            mask_01 = [("valu", ("*", b['idx'][g], b['idx'][g], b['tmp1'][g])) for g in [0, 1]]
            ops_list.append(bounds_47 + mask_01)
            # Cycle 8: multiply_mask g2-7 (6 ops)
            ops_list.append([("valu", ("*", b['idx'][g], b['idx'][g], b['tmp1'][g])) for g in range(2, 8)])
            return ops_list

        def build_gather_ops(b):
            """Build gather ops for batch b, 2 loads per cycle"""
            ops_list = []
            for g in range(BATCH_SIZE):
                for i in range(0, VLEN, 2):
                    ops_list.append([
                        ("load", ("load", b['node'][g] + i, b['ga'][g][i])),
                        ("load", ("load", b['node'][g] + i + 1, b['ga'][g][i + 1])),
                    ])
            return ops_list

        # Precompute group offset constants for all batches
        batch_g_consts = []
        for batch_idx in range(n_batches):
            base_group = batch_idx * BATCH_SIZE
            batch_g_consts.append([self.scratch_const((base_group + g) * VLEN) for g in range(BATCH_SIZE)])

        def build_addr_alu_ops(b, batch_idx):
            """Build ALU ops for address computation (can run during valu)"""
            g_consts = batch_g_consts[batch_idx]
            ops_list = []
            for alu_batch in range(0, BATCH_SIZE, 6):
                ops = []
                for g in range(alu_batch, min(alu_batch + 6, BATCH_SIZE)):
                    ops.append(("alu", ("+", b['addr_idx'][g], self.scratch["inp_indices_p"], g_consts[g])))
                    ops.append(("alu", ("+", b['addr_val'][g], self.scratch["inp_values_p"], g_consts[g])))
                ops_list.append(ops)
            return ops_list

        def build_vload_ops(b):
            """Build vload ops (uses load slots, conflicts with gather)"""
            ops_list = []
            for g in range(BATCH_SIZE):
                ops_list.append([
                    ("load", ("vload", b['idx'][g], b['addr_idx'][g])),
                    ("load", ("vload", b['val'][g], b['addr_val'][g])),
                ])
            return ops_list

        def build_ga_alu_ops(b):
            """Build ALU ops for gather address computation (can run during valu)"""
            ops_queue = []
            for g in range(BATCH_SIZE):
                for i in range(VLEN):
                    ops_queue.append(("alu", ("+", b['ga'][g][i], self.scratch["forest_values_p"], b['idx'][g] + i)))
            ops_list = []
            for i in range(0, len(ops_queue), 12):
                ops_list.append(ops_queue[i:i+12])
            return ops_list

        def emit_load_phase_sequential(b, batch_idx):
            """Emit address computation and vload with careful dependency handling"""
            addr_ops = build_addr_alu_ops(b, batch_idx)  # 2 cycles: g0-5, g6-7
            vload_ops = build_vload_ops(b)               # 8 cycles: g0, g1, ..., g7

            # Cycle 1: addr g0-5 (no vload - addr not ready)
            self.instrs.append(self.build_vliw(addr_ops[0]))
            # Cycle 2: addr g6-7 + vload g0 (g0 addr computed in cycle 1)
            if len(addr_ops) > 1:
                self.instrs.append(self.build_vliw(addr_ops[1] + vload_ops[0]))
            else:
                self.instrs.append(self.build_vliw(vload_ops[0]))
            # Cycles 3-9: vload g1-g7 (g1-5 addr ready from cycle 1, g6-7 ready from cycle 2)
            for ops in vload_ops[1:]:
                self.instrs.append(self.build_vliw(ops))

            ga_ops = build_ga_alu_ops(b)
            for ops in ga_ops:
                self.instrs.append(self.build_vliw(ops))

        def emit_store_phase(b):
            """Emit store instructions for a batch"""
            for g in range(BATCH_SIZE):
                self.instrs.append(self.build_vliw([
                    ("store", ("vstore", b['addr_idx'][g], b['idx'][g])),
                    ("store", ("vstore", b['addr_val'][g], b['val'][g])),
                ]))

        def build_store_ops(b):
            """Build store ops, 2 vstores per cycle = 8 cycles for 8 groups"""
            ops_list = []
            for g in range(BATCH_SIZE):
                ops_list.append([
                    ("store", ("vstore", b['addr_idx'][g], b['idx'][g])),
                    ("store", ("vstore", b['addr_val'][g], b['val'][g])),
                ])
            return ops_list

        # CROSS-ROUND PIPELINING: Use round-offset buffer indexing
        # 4 buffers for 4 batches: (i-2)%4 ≠ (i+1)%4 always (diff=3)
        # This enables overlapping addr_ALU[i+1] with store[i-2] in steady state!
        def get_buf(batch_idx, round_idx):
            return buf[(batch_idx + round_idx) % 4]

        for round_idx in range(rounds):
            buf_0 = get_buf(0, round_idx)
            buf_1 = get_buf(1, round_idx)
            buf_2 = get_buf(2, round_idx)

            if round_idx == 0:
                # First round: full prologue for batch 0
                emit_load_phase_sequential(buf_0, 0)
            else:
                # Later rounds: addr_ALU + vload done in prev epilogue, just do ga_ALU
                ga_ops_0 = build_ga_alu_ops(buf_0)
                for ops in ga_ops_0:
                    self.instrs.append(self.build_vliw(ops))

            # gather[0] + early addr_ALU[1]
            gather_ops_0 = build_gather_ops(buf_0)
            addr_ops_1 = build_addr_alu_ops(buf_1, 1)
            for i, cycle_ops in enumerate(gather_ops_0):
                bundle = list(cycle_ops)
                if i < len(addr_ops_1):
                    bundle.extend(addr_ops_1[i])
                self.instrs.append(self.build_vliw(bundle))

            # Prologue 2: skip addr_ALU[1] (done above), do vload[1], ga_ALU[1]
            vload_ops_1 = build_vload_ops(buf_1)
            for ops in vload_ops_1:
                self.instrs.append(self.build_vliw(ops))
            ga_ops_1 = build_ga_alu_ops(buf_1)
            for ops in ga_ops_1:
                self.instrs.append(self.build_vliw(ops))

            # gather[1] + valu[0] + early addr_ALU[2]
            gather_ops_1 = build_gather_ops(buf_1)
            valu_ops_0 = build_valu_ops(buf_0)
            addr_ops_2 = build_addr_alu_ops(buf_2, 2)
            g_idx, v_idx, a_idx = 0, 0, 0
            while g_idx < len(gather_ops_1) or v_idx < len(valu_ops_0):
                bundle = []
                if g_idx < len(gather_ops_1):
                    bundle.extend(gather_ops_1[g_idx])
                    g_idx += 1
                if v_idx < len(valu_ops_0):
                    bundle.extend(valu_ops_0[v_idx])
                    v_idx += 1
                if a_idx < len(addr_ops_2):
                    bundle.extend(addr_ops_2[a_idx])
                    a_idx += 1
                self.instrs.append(self.build_vliw(bundle))

            # Steady state: batches 2, 3, ...
            for batch_idx in range(2, n_batches):
                cur_buf = get_buf(batch_idx, round_idx)
                prev_buf = get_buf(batch_idx - 1, round_idx)
                prev_prev_buf = get_buf(batch_idx - 2, round_idx)

                # addr_ALU[2] was done in prologue 2's gather+valu phase
                # For batch 3+, overlap addr_ALU with valu (ALU vs VALU slots, different buffers)
                vload_ops = build_vload_ops(cur_buf)
                valu_ops = build_valu_ops(prev_buf)
                v_idx = 0

                if batch_idx > 2:
                    # Overlap addr_ALU[i] + valu[i-1], then vload[i] + valu[i-1]
                    # Stagger: cycle 0 = addr[g0-5]+valu, cycle 1 = addr[g6-7]+vload[g0]+valu
                    addr_ops = build_addr_alu_ops(cur_buf, batch_idx)
                    # Cycle 0: addr[g0-5] + valu
                    bundle = list(addr_ops[0])
                    if v_idx < len(valu_ops):
                        bundle.extend(valu_ops[v_idx])
                        v_idx += 1
                    self.instrs.append(self.build_vliw(bundle))
                    # Cycle 1: addr[g6-7] + vload[g0] + valu
                    bundle = list(addr_ops[1]) + list(vload_ops[0])
                    if v_idx < len(valu_ops):
                        bundle.extend(valu_ops[v_idx])
                        v_idx += 1
                    self.instrs.append(self.build_vliw(bundle))
                    # Cycles 2-8: vload[g1-7] + valu
                    for vl_ops in vload_ops[1:]:
                        bundle = list(vl_ops)
                        if v_idx < len(valu_ops):
                            bundle.extend(valu_ops[v_idx])
                            v_idx += 1
                        self.instrs.append(self.build_vliw(bundle))
                else:
                    # Batch 2: addr_ALU[2] already done, just vload + valu
                    for vl_ops in vload_ops:
                        bundle = list(vl_ops)
                        if v_idx < len(valu_ops):
                            bundle.extend(valu_ops[v_idx])
                            v_idx += 1
                        self.instrs.append(self.build_vliw(bundle))

                # Phase 2: ga_ALU[i] + valu[i-1] continuing - 6 cycles
                ga_ops = build_ga_alu_ops(cur_buf)
                for ga_cycle in ga_ops:
                    bundle = list(ga_cycle)
                    if v_idx < len(valu_ops):
                        bundle.extend(valu_ops[v_idx])
                        v_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

                # Phase 3: gather[i] + valu[i-1] remaining + store[i-2]
                gather_ops = build_gather_ops(cur_buf)
                store_ops = build_store_ops(prev_prev_buf)
                g_idx, s_idx = 0, 0
                while g_idx < len(gather_ops) or v_idx < len(valu_ops) or s_idx < len(store_ops):
                    bundle = []
                    if g_idx < len(gather_ops):
                        bundle.extend(gather_ops[g_idx])
                        g_idx += 1
                    if v_idx < len(valu_ops):
                        bundle.extend(valu_ops[v_idx])
                        v_idx += 1
                    if s_idx < len(store_ops):
                        bundle.extend(store_ops[s_idx])
                        s_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

            # Epilogue 1: valu[n_batches-1] + store[n_batches-2]
            last_buf = get_buf(n_batches - 1, round_idx)
            second_last_buf = get_buf(n_batches - 2, round_idx)
            valu_ops = build_valu_ops(last_buf)
            store_ops = build_store_ops(second_last_buf)
            v_idx, s_idx = 0, 0
            while v_idx < len(valu_ops) or s_idx < len(store_ops):
                bundle = []
                if v_idx < len(valu_ops):
                    bundle.extend(valu_ops[v_idx])
                    v_idx += 1
                if s_idx < len(store_ops):
                    bundle.extend(store_ops[s_idx])
                    s_idx += 1
                self.instrs.append(self.build_vliw(bundle))

            # Epilogue 2: store[last] + overlap with next round's start
            store_ops = build_store_ops(last_buf)
            if round_idx < rounds - 1:
                # Cross-round overlap: store uses buf[(3+R)%3], next round uses buf[(R+1)%3]
                # These are different, so we can overlap!
                next_buf_0 = get_buf(0, round_idx + 1)
                next_addr_ops = build_addr_alu_ops(next_buf_0, 0)
                next_vload_ops = build_vload_ops(next_buf_0)

                # Optimized interleave: start vload[g0] in cycle 2 (after addr[g0-5] in cycle 1)
                # Cycle 1: store + addr[g0-5]
                bundle = list(store_ops[0]) + list(next_addr_ops[0])
                self.instrs.append(self.build_vliw(bundle))
                # Cycle 2: store + addr[g6-7] + vload[g0]
                bundle = list(store_ops[1]) + list(next_addr_ops[1]) + list(next_vload_ops[0])
                self.instrs.append(self.build_vliw(bundle))
                # Cycles 3-8: store + vload[g1-6]
                for i in range(2, 8):
                    bundle = list(store_ops[i]) + list(next_vload_ops[i - 1])
                    self.instrs.append(self.build_vliw(bundle))
                # Cycle 9: vload[g7] (store finished)
                self.instrs.append(self.build_vliw(next_vload_ops[7]))
            else:
                # Last round - just store
                for cycle_ops in store_ops:
                    self.instrs.append(self.build_vliw(cycle_ops))

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
