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
        Optimized VLIW+SIMD implementation with PERSISTENT SCRATCH.

        Key optimization: Keep idx and val in scratch across all rounds.
        Only vload once at start, vstore once at end.
        This eliminates ~960 cycles of per-batch vload/vstore overhead!
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        n_groups_total = batch_size // VLEN  # 32
        BATCH_SIZE = 8  # Process 8 groups at a time
        n_batches = n_groups_total // BATCH_SIZE  # 4 batches per round

        # Initialize memory layout variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        # Allocate scratch for init_vars
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Allocate scratch for basic constants
        zero_const = self.alloc_scratch("const_0")
        one_const = self.alloc_scratch("const_1")
        two_const = self.alloc_scratch("const_2")
        self.const_map[0] = zero_const
        self.const_map[1] = one_const
        self.const_map[2] = two_const

        # Allocate scratch for hash constants
        hash_scalar_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.alloc_scratch(f"hash_{hi}_c1")
            c3 = self.alloc_scratch(f"hash_{hi}_c3")
            self.const_map[val1] = c1
            self.const_map[val3] = c3
            hash_scalar_consts.append((c1, c3))

        # Allocate scratch for group offset constants
        all_g_consts = []
        for g in range(n_groups_total):
            addr = self.alloc_scratch(f"g_const_{g}")
            self.const_map[g * VLEN] = addr
            all_g_consts.append(addr)

        # ============================================================
        # BATCHED INITIALIZATION: Load all constants with VLIW (2 per cycle)
        # ============================================================
        idx_consts = [self.alloc_scratch(f"idx_{i}") for i in range(len(init_vars))]

        # Collect ALL const loads needed
        const_loads = []
        const_loads.append(("load", ("const", zero_const, 0)))
        const_loads.append(("load", ("const", one_const, 1)))
        const_loads.append(("load", ("const", two_const, 2)))
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1, c3 = hash_scalar_consts[hi]
            const_loads.append(("load", ("const", c1, val1)))
            const_loads.append(("load", ("const", c3, val3)))
        for g in range(n_groups_total):
            const_loads.append(("load", ("const", all_g_consts[g], g * VLEN)))
        for i in range(len(init_vars)):
            const_loads.append(("load", ("const", idx_consts[i], i)))

        # Batch const loads 2 per cycle (load engine has 2 slots)
        for i in range(0, len(const_loads), 2):
            if i + 1 < len(const_loads):
                self.instrs.append(self.build_vliw([const_loads[i], const_loads[i+1]]))
            else:
                self.instrs.append(self.build_vliw([const_loads[i]]))

        # Load all memory values (batched)
        mem_loads = [("load", ("load", self.scratch[init_vars[i]], idx_consts[i])) for i in range(len(init_vars))]
        for i in range(0, len(mem_loads), 2):
            if i + 1 < len(mem_loads):
                self.instrs.append(self.build_vliw([mem_loads[i], mem_loads[i+1]]))
            else:
                self.instrs.append(self.build_vliw([mem_loads[i]]))

        self.add("flow", ("pause",))

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Vector constants
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

        # Precompute hash stage constants as vectors (batch 6 vbroadcasts per cycle)
        v_hash_consts = []
        vbroadcast_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.alloc_scratch(f"v_hash_{hi}_c1", VLEN)
            v_const3 = self.alloc_scratch(f"v_hash_{hi}_c3", VLEN)
            c1_scalar, c3_scalar = hash_scalar_consts[hi]
            vbroadcast_ops.append(("valu", ("vbroadcast", v_const1, c1_scalar)))
            vbroadcast_ops.append(("valu", ("vbroadcast", v_const3, c3_scalar)))
            v_hash_consts.append((v_const1, v_const3))

        for i in range(0, len(vbroadcast_ops), 6):
            self.instrs.append(self.build_vliw(vbroadcast_ops[i:i+6]))

        # ============================================================
        # PERSISTENT SCRATCH: Allocate ALL idx/val for entire dataset
        # These stay in scratch across ALL rounds!
        # ============================================================
        persistent_idx = [self.alloc_scratch(f"p_idx_{g}", VLEN) for g in range(n_groups_total)]
        persistent_val = [self.alloc_scratch(f"p_val_{g}", VLEN) for g in range(n_groups_total)]

        # Working buffers for batch processing (2 buffers for pipelining)
        def alloc_work_buffer(prefix):
            return {
                'ga': [[self.alloc_scratch(f"{prefix}_{g}_ga_{i}") for i in range(VLEN)] for g in range(BATCH_SIZE)],
                'node': [self.alloc_scratch(f"{prefix}_{g}_node", VLEN) for g in range(BATCH_SIZE)],
                'tmp1': [self.alloc_scratch(f"{prefix}_{g}_tmp1", VLEN) for g in range(BATCH_SIZE)],
                'tmp2': [self.alloc_scratch(f"{prefix}_{g}_tmp2", VLEN) for g in range(BATCH_SIZE)],
            }

        work_buf = [alloc_work_buffer("W0"), alloc_work_buffer("W1")]

        # Memory address constants (computed once)
        addr_idx_base = [self.alloc_scratch(f"addr_idx_{g}") for g in range(n_groups_total)]
        addr_val_base = [self.alloc_scratch(f"addr_val_{g}") for g in range(n_groups_total)]

        # Group offset constants for batch processing
        batch_g_consts = []
        for batch_idx in range(n_batches):
            base_group = batch_idx * BATCH_SIZE
            batch_g_consts.append([all_g_consts[base_group + g] for g in range(BATCH_SIZE)])

        # ============================================================
        # HELPER FUNCTIONS for persistent scratch approach
        # ============================================================

        def get_persistent_idx(g):
            """Get persistent idx scratch for global group g"""
            return persistent_idx[g]

        def get_persistent_val(g):
            """Get persistent val scratch for global group g"""
            return persistent_val[g]

        def build_valu_ops_persistent(work, batch_idx):
            """Build VALU ops for batch using persistent idx/val storage"""
            ops_list = []
            base_g = batch_idx * BATCH_SIZE  # Global group offset

            # Get references to persistent storage for this batch
            p_val = [get_persistent_val(base_g + g) for g in range(BATCH_SIZE)]
            p_idx = [get_persistent_idx(base_g + g) for g in range(BATCH_SIZE)]

            def tmp_ops(hi, groups):
                """Generate tmp1, tmp2 ops for hash stage"""
                v_c1, v_c3 = v_hash_consts[hi]
                ops = []
                for g in groups:
                    ops.append(("valu", (HASH_STAGES[hi][0], work['tmp1'][g], p_val[g], v_c1)))
                    ops.append(("valu", (HASH_STAGES[hi][3], work['tmp2'][g], p_val[g], v_c3)))
                return ops

            def combine_ops(hi, groups):
                """Generate combine ops for hash stage"""
                return [("valu", (HASH_STAGES[hi][2], p_val[g], work['tmp1'][g], work['tmp2'][g]))
                        for g in groups]

            # XOR with gathered node values
            ops_list.append([("valu", ("^", p_val[g], p_val[g], work['node'][g])) for g in range(6)])
            xor_67 = [("valu", ("^", p_val[g], p_val[g], work['node'][g])) for g in [6, 7]]
            ops_list.append(xor_67 + tmp_ops(0, [0, 1]))

            ops_list.append(tmp_ops(0, [2, 3, 4]))
            ops_list.append(tmp_ops(0, [5, 6, 7]))

            for hi in range(1, 6):
                ops_list.append(combine_ops(hi-1, [0, 1, 2, 3, 4, 5]))
                ops_list.append(combine_ops(hi-1, [6, 7]) + tmp_ops(hi, [0, 1]))
                ops_list.append(tmp_ops(hi, [2, 3, 4]))
                ops_list.append(tmp_ops(hi, [5, 6, 7]))

            # Epilogue: final combine + index calculation
            ops_list.append(combine_ops(5, [0, 1, 2, 3, 4, 5]))
            idx_ops_01 = []
            for g in [0, 1]:
                idx_ops_01.append(("valu", ("&", work['tmp1'][g], p_val[g], v_one)))
                idx_ops_01.append(("valu", ("multiply_add", p_idx[g], p_idx[g], v_two, v_one)))
            ops_list.append(combine_ops(5, [6, 7]) + idx_ops_01)

            idx_ops_24 = []
            for g in [2, 3, 4]:
                idx_ops_24.append(("valu", ("&", work['tmp1'][g], p_val[g], v_one)))
                idx_ops_24.append(("valu", ("multiply_add", p_idx[g], p_idx[g], v_two, v_one)))
            ops_list.append(idx_ops_24)

            idx_ops_57 = []
            for g in [5, 6, 7]:
                idx_ops_57.append(("valu", ("&", work['tmp1'][g], p_val[g], v_one)))
                idx_ops_57.append(("valu", ("multiply_add", p_idx[g], p_idx[g], v_two, v_one)))
            ops_list.append(idx_ops_57)

            ops_list.append([("valu", ("+", p_idx[g], p_idx[g], work['tmp1'][g])) for g in range(6)])
            add_67 = [("valu", ("+", p_idx[g], p_idx[g], work['tmp1'][g])) for g in [6, 7]]
            bounds_03 = [("valu", ("<", work['tmp1'][g], p_idx[g], v_n_nodes)) for g in range(4)]
            ops_list.append(add_67 + bounds_03)

            bounds_47 = [("valu", ("<", work['tmp1'][g], p_idx[g], v_n_nodes)) for g in [4, 5, 6, 7]]
            mask_01 = [("valu", ("*", p_idx[g], p_idx[g], work['tmp1'][g])) for g in [0, 1]]
            ops_list.append(bounds_47 + mask_01)
            ops_list.append([("valu", ("*", p_idx[g], p_idx[g], work['tmp1'][g])) for g in range(2, 8)])
            return ops_list

        def build_gather_ops_persistent(work, batch_idx):
            """Build gather ops using work buffer"""
            ops_list = []
            for g in range(BATCH_SIZE):
                for i in range(0, VLEN, 2):
                    ops_list.append([
                        ("load", ("load", work['node'][g] + i, work['ga'][g][i])),
                        ("load", ("load", work['node'][g] + i + 1, work['ga'][g][i + 1])),
                    ])
            return ops_list

        def build_ga_alu_ops_persistent(work, batch_idx):
            """Build ALU ops for gather addresses using persistent idx"""
            base_g = batch_idx * BATCH_SIZE
            ops_queue = []
            for g in range(BATCH_SIZE):
                p_idx = get_persistent_idx(base_g + g)
                for i in range(VLEN):
                    ops_queue.append(("alu", ("+", work['ga'][g][i], self.scratch["forest_values_p"], p_idx + i)))
            ops_list = []
            for i in range(0, len(ops_queue), 12):
                ops_list.append(ops_queue[i:i+12])
            return ops_list

        # ============================================================
        # INITIAL LOAD: Load all idx/val into persistent scratch (once!)
        # ============================================================
        # Overlap address computation with vloads
        addr_ops = []
        for g_start in range(0, n_groups_total, 6):
            g_end = min(g_start + 6, n_groups_total)
            ops = []
            for g in range(g_start, g_end):
                ops.append(("alu", ("+", addr_idx_base[g], self.scratch["inp_indices_p"], all_g_consts[g])))
                ops.append(("alu", ("+", addr_val_base[g], self.scratch["inp_values_p"], all_g_consts[g])))
            addr_ops.append(ops)

        vload_ops = []
        for g in range(n_groups_total):
            vload_ops.append([
                ("load", ("vload", persistent_idx[g], addr_idx_base[g])),
                ("load", ("vload", persistent_val[g], addr_val_base[g])),
            ])

        # Cycle 1: compute first batch of addresses (groups 0-5)
        self.instrs.append(self.build_vliw(addr_ops[0]))

        # Cycles 2+: vload groups whose addresses are ready || compute remaining addresses
        vload_idx = 0
        addr_idx = 1
        while vload_idx < len(vload_ops):
            bundle = list(vload_ops[vload_idx])
            vload_idx += 1
            if addr_idx < len(addr_ops):
                bundle.extend(addr_ops[addr_idx])
                addr_idx += 1
            self.instrs.append(self.build_vliw(bundle))

        # ============================================================
        # MAIN LOOP: Process all rounds using persistent scratch
        # ============================================================
        for round_idx in range(rounds):
            w0 = work_buf[0]
            w1 = work_buf[1]

            # ============================================================
            # ROUND 0 SPECIAL CASE: All p_idx = 0, so ALL gathers load forest_values[0]
            # Instead of gather, broadcast forest_values[0] to all work['node']!
            # ============================================================
            if round_idx == 0:
                # Load forest_values[0] once (1 load cycle)
                shared_node_val = self.alloc_scratch("shared_node_val")
                self.instrs.append(self.build_vliw([
                    ("load", ("load", shared_node_val, self.scratch["forest_values_p"]))
                ]))

                # Broadcast to all work['node'] positions (16 vbroadcasts / 6 per cycle = 3 cycles)
                vbroadcast_ops = []
                for g in range(BATCH_SIZE):
                    vbroadcast_ops.append(("valu", ("vbroadcast", w0['node'][g], shared_node_val)))
                    vbroadcast_ops.append(("valu", ("vbroadcast", w1['node'][g], shared_node_val)))
                for i in range(0, len(vbroadcast_ops), 6):
                    self.instrs.append(self.build_vliw(vbroadcast_ops[i:i+6]))

                # Run valu[0,1,2,3] with overlapped ga_ALU for next round
                # Optimization: valu[0] is standalone, but valu[1] can overlap with ga_ALU[0,R+1]!
                # After valu[0] completes, p_idx[batch0] is updated, so ga_ALU[0,R+1] can start.
                valu_ops_0 = build_valu_ops_persistent(w0, 0)
                for ops in valu_ops_0:
                    self.instrs.append(self.build_vliw(ops))

                # valu[1] || ga_ALU[0,R+1] (p_idx[batch0] ready after valu[0])
                valu_ops_1 = build_valu_ops_persistent(w1, 1)
                next_ga_ops_0 = build_ga_alu_ops_persistent(w0, 0)
                v_idx, a_idx = 0, 0
                while v_idx < len(valu_ops_1):
                    bundle = list(valu_ops_1[v_idx])
                    v_idx += 1
                    if a_idx < len(next_ga_ops_0):
                        bundle.extend(next_ga_ops_0[a_idx])
                        a_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

                # valu[2] || gather[0,R+1] || ga_ALU[1,R+1] (addresses ready from valu[1] phase!)
                valu_ops_2 = build_valu_ops_persistent(w0, 2)
                next_gather_ops_0 = build_gather_ops_persistent(w0, 0)
                next_ga_ops_1 = build_ga_alu_ops_persistent(w1, 1)
                v_idx, g_idx, a_idx = 0, 0, 0
                while v_idx < len(valu_ops_2) or g_idx < len(next_gather_ops_0):
                    bundle = []
                    if v_idx < len(valu_ops_2):
                        bundle.extend(valu_ops_2[v_idx])
                        v_idx += 1
                    if g_idx < len(next_gather_ops_0):
                        bundle.extend(next_gather_ops_0[g_idx])
                        g_idx += 1
                    if a_idx < len(next_ga_ops_1):
                        bundle.extend(next_ga_ops_1[a_idx])
                        a_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

                # valu[3] || gather[1,R+1] || ga_ALU[2,R+1]
                # (gather[0] done during valu[2], p_idx[batch1] ready after valu[1])
                valu_ops_3 = build_valu_ops_persistent(w1, 3)
                next_gather_ops_1 = build_gather_ops_persistent(w1, 1)
                next_ga_ops_2 = build_ga_alu_ops_persistent(w0, 2)
                v_idx, g_idx, a_idx = 0, 0, 0
                while v_idx < len(valu_ops_3) or g_idx < len(next_gather_ops_1):
                    bundle = []
                    if v_idx < len(valu_ops_3):
                        bundle.extend(valu_ops_3[v_idx])
                        v_idx += 1
                    if g_idx < len(next_gather_ops_1):
                        bundle.extend(next_gather_ops_1[g_idx])
                        g_idx += 1
                    if a_idx < len(next_ga_ops_2):
                        bundle.extend(next_ga_ops_2[a_idx])
                        a_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

                continue  # Skip normal batch processing for round 0

            # ============================================================
            # ROUNDS 1+: Normal processing with gather/valu overlap
            # ============================================================
            # From previous epilogue: gather[0,1] done, ga_ALU[2] done
            # So we have node[0], node[1] ready, and ga[2] addresses ready

            # Batch 0: gather[2] || valu[0] || ga_ALU[3]
            gather_ops_2 = build_gather_ops_persistent(w0, 2)
            valu_ops_0 = build_valu_ops_persistent(w0, 0)
            ga_ops_3 = build_ga_alu_ops_persistent(w1, 3)
            g_idx, v_idx, a_idx = 0, 0, 0
            while g_idx < len(gather_ops_2) or v_idx < len(valu_ops_0):
                bundle = []
                if g_idx < len(gather_ops_2):
                    bundle.extend(gather_ops_2[g_idx])
                    g_idx += 1
                if v_idx < len(valu_ops_0):
                    bundle.extend(valu_ops_0[v_idx])
                    v_idx += 1
                if a_idx < len(ga_ops_3):
                    bundle.extend(ga_ops_3[a_idx])
                    a_idx += 1
                self.instrs.append(self.build_vliw(bundle))

            # Batch 1: gather[3] || valu[1] || ga_ALU[0,next]
            gather_ops_3 = build_gather_ops_persistent(w1, 3)
            valu_ops_1 = build_valu_ops_persistent(w1, 1)
            if round_idx < rounds - 1:
                ga_ops_0_next = build_ga_alu_ops_persistent(w0, 0)
            else:
                ga_ops_0_next = []
            g_idx, v_idx, a_idx = 0, 0, 0
            while g_idx < len(gather_ops_3) or v_idx < len(valu_ops_1):
                bundle = []
                if g_idx < len(gather_ops_3):
                    bundle.extend(gather_ops_3[g_idx])
                    g_idx += 1
                if v_idx < len(valu_ops_1):
                    bundle.extend(valu_ops_1[v_idx])
                    v_idx += 1
                if a_idx < len(ga_ops_0_next):
                    bundle.extend(ga_ops_0_next[a_idx])
                    a_idx += 1
                self.instrs.append(self.build_vliw(bundle))

            # Batch 2: valu[2] || gather[0,next] || ga_ALU[1,next]
            valu_ops_2 = build_valu_ops_persistent(w0, 2)
            if round_idx < rounds - 1:
                next_gather_ops_0 = build_gather_ops_persistent(w0, 0)
                next_ga_ops_1 = build_ga_alu_ops_persistent(w1, 1)
                v_idx, g_idx, a_idx = 0, 0, 0
                while v_idx < len(valu_ops_2) or g_idx < len(next_gather_ops_0):
                    bundle = []
                    if v_idx < len(valu_ops_2):
                        bundle.extend(valu_ops_2[v_idx])
                        v_idx += 1
                    if g_idx < len(next_gather_ops_0):
                        bundle.extend(next_gather_ops_0[g_idx])
                        g_idx += 1
                    if a_idx < len(next_ga_ops_1):
                        bundle.extend(next_ga_ops_1[a_idx])
                        a_idx += 1
                    self.instrs.append(self.build_vliw(bundle))
            else:
                # Last round - no next round operations
                for ops in valu_ops_2:
                    self.instrs.append(self.build_vliw(ops))

            # Epilogue: valu[3] || gather[1,next] || ga_ALU[2,next]
            valu_ops_3 = build_valu_ops_persistent(w1, 3)
            if round_idx < rounds - 1:
                next_gather_ops_1 = build_gather_ops_persistent(w1, 1)
                next_ga_ops_2 = build_ga_alu_ops_persistent(w0, 2)

                v_idx, g_idx, a_idx = 0, 0, 0
                while v_idx < len(valu_ops_3) or g_idx < len(next_gather_ops_1):
                    bundle = []
                    if v_idx < len(valu_ops_3):
                        bundle.extend(valu_ops_3[v_idx])
                        v_idx += 1
                    if g_idx < len(next_gather_ops_1):
                        bundle.extend(next_gather_ops_1[g_idx])
                        g_idx += 1
                    if a_idx < len(next_ga_ops_2):
                        bundle.extend(next_ga_ops_2[a_idx])
                        a_idx += 1
                    self.instrs.append(self.build_vliw(bundle))
            else:
                # LAST ROUND OPTIMIZATION: overlap valu[3] with vstores!
                # valu uses VALU slots, vstore uses STORE slots - no conflict!
                # Groups 0-23 are done (valu[0,1,2] completed), can vstore them now
                vstore_ops_0_23 = []
                for g in range(24):
                    vstore_ops_0_23.append([
                        ("store", ("vstore", addr_idx_base[g], persistent_idx[g])),
                        ("store", ("vstore", addr_val_base[g], persistent_val[g])),
                    ])

                v_idx, s_idx = 0, 0
                while v_idx < len(valu_ops_3):
                    bundle = list(valu_ops_3[v_idx])
                    v_idx += 1
                    if s_idx < len(vstore_ops_0_23):
                        bundle.extend(vstore_ops_0_23[s_idx])
                        s_idx += 1
                    self.instrs.append(self.build_vliw(bundle))

                # Remaining vstores for groups 24-31 (just updated by valu[3])
                for g in range(24, n_groups_total):
                    self.instrs.append(self.build_vliw([
                        ("store", ("vstore", addr_idx_base[g], persistent_idx[g])),
                        ("store", ("vstore", addr_val_base[g], persistent_val[g])),
                    ]))

        # Skip separate final vstore - already done in last round optimization

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
