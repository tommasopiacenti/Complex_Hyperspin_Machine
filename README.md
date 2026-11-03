# Complex_HSM v2.9 — Comprehensive Reference

*Last Updated: 18/09/2026 | Author: T. Piacenti*

---

## Acknowledgments
Funded by the European Union (HORIZON-ERC-2023-ADG HYPERSPIM project Grant No. 101139828)
[hyperpsim.org](https://hyperspim.newcomplexlight.org)

---

## Table of Contents

1. [Overview & Quick Start](#1-overview--quick-start)
2. [Physics Model](#2-physics-model)
3. [Numerical Integration](#3-numerical-integration)
4. [Code Architecture](#4-code-architecture)
5. [Sparse-First Architecture](#5-sparse-first-architecture)
6. [Coupling Matrix Generation](#6-coupling-matrix-generation)
7. [Configuration Reference](#7-configuration-reference)
8. [Output Files](#8-output-files)
9. [Memory Scaling Analysis](#9-memory-scaling-analysis)
10. [Parallelization](#10-parallelization)
11. [Scripts & Workflows](#11-scripts--workflows)
12. [Complete Function Reference](#12-complete-function-reference)
13. [Troubleshooting & FAQ](#13-troubleshooting--faq)

---

## 1. Overview & Quick Start

Complex_HSM simulates the dynamics of a system of parametric oscillators used as a **Complex Hyperspin Machine**. The natural dynamical variables are N×D oscillator fields arranged as **N complex hyperspin vectors** S_a ∈ ℂ^D (a = 0, ..., N−1), each with D components, evolving via second-order Runge-Kutta (RK2) integration of driven-dissipative ODEs. The hyperspin machine is connected to an equalization layer, composed of 2N **equalization fields**, which consists of two equalization rings: The parametric oscillator fields in each couple of neighboring hyperspins is coupled to two equalization fields (Y0, Y1) that, for sufficiently long integration time, equalize the two hyperspin amplitudes. Consequently, amplitude equalization is achieved for all hyperspins.

The machine maps to a **hyperspin Hamiltonian** H — an energy functional defined on the *directions* of the fields, which we identify with complex hyperspins:

```
Ŝ_a = S_a / |S_a|       (unit-normalized field = hyperspin)
H = H({Ŝ_a})            (depends only on hyperspin configuration, not field magnitudes)
```

The dynamics drives the fields toward configurations that minimize H. A more negative H corresponds to a more stable hyperspin configuration; the global minimum of H is the **ground state**.

The **coupling matrix** J (N×N, Hermitian) defines pairwise interactions. It can be generated as a **random graph** (`new_coupling=1`), a **d-dimensional hypercubic lattice** (`new_coupling=2`), or **loaded from a file** (`new_coupling=0`). Output is available in both a sparse edge-list format (`Data_Matrix_sparse.dat`) and a dense N×N grid (`Data_Matrix.dat`), depending on the `compute_eigenvalues` setting.

### 1.1 Compile

```bash
gcc -O3 -fopenmp -march=native src/Complex_HSM.c -o Complex_HSM.x -lgsl -lgslcblas -lm
```

| Flag | Purpose |
|------|---------|
| `-O3` | Enables aggressive compiler optimizations: function inlining, loop vectorization, instruction-level parallelism. Substantial speedup on the RK2 inner loops. |
| `-fopenmp` | Enables OpenMP pragmas for multi-threaded execution. Without this flag the code compiles but runs single-threaded regardless of `num_threads` in the config. |
| `-march=native` | Generates instructions for the host CPU (e.g., AVX2, FMA). Typically 10–20% faster than generic x86-64 on the numerical loops. |
| `-lgsl -lgslcblas -lm` | Links GSL (eigenvalue computation for pump threshold and diagnostics), CBLAS (internal GSL dependency), and the math library. |

### 1.2 Run

```bash
./Complex_HSM.x <NS> <pth> Complex_HSM.ini
```

| Argument | Description |
|----------|-------------|
| `NS` | Random seed (integer). Controls all randomness: initial conditions, coupling matrix (if generated), pump values. Same NS + same config → identical results. |
| `pth` | Pump threshold fraction. `-1` = use fixed `hm` from config. `≥ 0` = auto-calc: `hm = pth × hth` (requires `compute_eigenvalues=1`). |
| `Complex_HSM.ini` | Configuration file (see §7 for full reference). |

### 1.3 Example: Sparse Random Graph

This walkthrough shows a run for a system of 100 2-dimensional hyperspins on a sparse random graph with complex coupling, zero pump, and observes the system evolve toward a steady state.

**Step 1 — Create the config file.** Save the following as `Complex_HSM.ini`:

```ini
# System
N   = 100
D   = 2

# Coupling: random graph, 10% edge probability, equal FM/AFM mix
new_coupling    = 1
prob_coup       = 0.1
prob_pm_re     = 0.5
prob_pm_im     = 0.5
coup_phase     = 0
type_coup       = 2
compute_eigenvalues = 0

# Dynamics: no pump, moderate dissipation and nonlinearity
hm    = 0
gm    = 0
beta  = 0.1
deq   = 1.0
dt    = 0.01
tmax  = 5000
t_coup = 0
t_eq   = 0

# Output: merged observables, write every 10 steps
save_dyn    = 3
save_steps  = 10
num_threads = 8
```

Key parameters for this example:
- `new_coupling=1`: generate a random graph (alternatives: `2` for lattice, `0` to load from file)
- `type_coup=2`: complex coupling (both Re and Im parts; `0`=real only, `1`=imag only)
- `compute_eigenvalues=0`: sparse-only mode (no dense J allocation — used for large systems; set to `1` to enable pump threshold auto-calculation and eigenvalue diagnostics)
- `save_dyn=3`: write merged observables to a single file (alternatives: `0`=final state only, `1`=full field dynamics, `2`=H+Lyapunov)
- `save_steps=10`: compute and write observables every 10th step

**Step 2 — Compile and run:**

```bash
gcc -O3 -fopenmp -march=native src/Complex_HSM.c -o Complex_HSM.x -lgsl -lgslcblas -lm
./Complex_HSM.x 1234 -1 Complex_HSM.ini
```

Here `1234` is the random seed (NS) and `-1` means use `hm=0` directly.

**Step 3 — Output.** The program writes:

- `Data_H_Ly_m_DS2.dat` — time series of observables (5 columns, one row per output step):

```
t           H            Ly/N         <m>          <DS2>
0.000000000000000  -0.023456789012345  -0.000123456789012  0.045678901234567  0.000012345678901
0.100000000000000  -0.034567890123456  -0.000234567890123  0.056789012345678  0.000023456789012
...
```

  - **t**: simulation time
  - **H**: hyperspin Hamiltonian (total energy; more negative = more stable hyperspin configuration)
  - **Ly/N**: Lyapunov exponent per field (>0 = chaotic/amplified, <0 = stable)
  - **⟨m⟩**: average magnetization (0 = disordered, 1 = fully aligned)
  - **⟨DS²⟩**: amplitude variance (measures uniformity of field magnitudes)

- `Data_Final.dat` — final field configuration (one line: seed, time, then |S|² and Re/Im components for each field)

**Step 4 — Matrix files.** The coupling matrix is saved automatically:

- `Data_Matrix_sparse.dat` — sparse edge-list format: edge count on the first line, then one edge per line (`a  b  Jr  Ji` where a < b). This is always written.
- `Data_Matrix.dat` — dense N×N grid format (only written when `compute_eigenvalues=1`).

Both files are human-readable. See §6.4–6.5 for format details.

### 1.4 Parallelism

Set `num_threads` in the config file:

- `num_threads = 1`: sequential execution, no OpenMP overhead
- `num_threads = 8`: parallel execution using 8 CPU cores

No recompilation needed — the thread count is set at runtime. See §10 for detailed guidance on when parallelization pays off.

---

## 2. Physics Model

### 2.1 System State

The system consists of **N interacting complex fields** S_a ∈ ℂ^D (a = 0, ..., N−1). The dynamics drives them toward configurations that minimize an energy functional defined on their *directions* — the **hyperspins** Ŝ_a = S_a/|S_a|.

```
S_a ∈ ℂ^D,    a = 0, ..., N-1
```

Each couple of neighboring hyperspin fields (a, a+1 mod N) in the equalization ring carries an **equalization field** Y_a ∈ ℂ² (components Y_{a,0}, Y_{a,1}) that stabilizes the amplitude difference across the couple:

```
Y_a ∈ ℂ^2,    a = 0, ..., N-1    (one per hyperspin-field couple in the ring; Y0/Y1 absorb opposite norm deviations)
```

The coupling is described by an N×N **Hermitian coupling matrix** J:

```
J_{ab} = Jr_{ab} + i·Ji_{ab},    with J^† = J
```

Hermiticity means Jr is symmetric (`Jr_{ab} = Jr_{ba}`) and Ji is anti-symmetric (`Ji_{ab} = -Ji_{ba}`). This ensures the coupling does not inject or remove net energy from the system — it only redistributes it.

### 2.2 Equations of Motion

Each field evolves according to a complex differential equation with three additive contributions:

```
dS_a/dt = f₀(S_a) + f_J(S_a) + f_eq(S_a)
```

#### Non-interacting term f₀ (drive + dissipation + nonlinearity)

The non-interacting dynamics of a single field under pump h_a, phase φ_a, loss g, and nonlinearity β. Implemented component-wise (real and imaginary parts separated):

```
Re(f₀) = ½·(½·h·cosφ − g − β·|S|²)·Re(S) + ½·h·sinφ·Im(S)
Im(f₀) = ½·(−½·h·cosφ − g − β·|S|²)·Im(S) + ½·h·sinφ·Re(S)
```

| Symbol | Meaning | Control parameter |
|--------|---------|-------------------|
| h_a | Pump amplitude for field a | `hm`, `dh` |
| φ_a | Pump phase for field a | `phim`, `dphi` |
| g | Intrinsic loss rate | `gm` |
| β | Nonlinear saturation strength | `beta` |

The cross-term proportional to sinφ couples Re(S) to Im(S) and vice versa — this is a non-Hermitian coupling between the real and imaginary components that enables amplification above threshold. When φ = 0, the sinφ term vanishes and the Re/Im evolution decouples (except through the coupling and equalization terms).

#### Coupling term f_J

```
f_J = ½·Σ_b J_{ab}·S_b
```

This is a matrix-vector product: the coupling field experienced by field a due to all other fields b. The factor of ½ is conventional to match the Hamiltonian definition (see §2.3). The Hermiticity of J guarantees that the coupling is conservative: it does not change the total Lyapunov function beyond what the non-interacting and equalization terms contribute.

#### Equalization term f_eq (amplitude stabilization)

The equalization variables live on interactions: for each pair of neighboring fields (a, b), there is an imbalance Y²_ab = |Y_{ab,0}|² − |Y_{ab,1}|² that measures their amplitude difference. The equalizers evolve with opposite signs:

```
dY_{ab,0}/dt = +½·deq·(|S_b|² − |S_a|²)·Y_{ab,0}
dY_{ab,1}/dt = −½·deq·(|S_b|² − |S_a|²)·Y_{ab,1}
```

When |S_b|² > |S_a|², Y²_ab grows. This feeds into the equalization force on each field as a restoring term:

```
f_eq = ½·deq·(Y²_{right} − Y²_{left})·S_a
```

that drives neighboring field norms toward each other. The equalizers act as **amplitude heterogeneity absorbers**: they communicate amplitude information across each interaction and adjust local gain/loss to equalize the amplitude distribution. Without equalization, fields can develop extreme amplitude disparities that lead to numerical instability.

### 2.3 Hamiltonian (Energy)

The total energy of the system, normalized by field magnitudes:

```
H = − Σ_{p,q} Σ_k [ Jr_{pq}·Ŝ_{pk}·conj(Ŝ_{qk}) + Ji_{pq}·Im(Ŝ_{pk}·conj(Ŝ_{qk})) ]
```

where `Ŝ_a = S_a / |S_a|` is the unit-normalized hyperspin. The normalization makes H **scale-invariant** — it depends only on the *direction* of each field, not its magnitude. This is essential because the physical dynamics allow field magnitudes to vary, but the energy of the steady state depends only on the orientational order.

A more negative H means a more stable configuration. The **ground state** (global minimum of H) is the configuration the system tries to reach through its dissipative dynamics.

**Implementation note:** The Hamiltonian is computed using the **edge list** (undirected pairs), with a ×2 factor to account for both (p,q) and (q,p) contributions. The `inv_sqrtS2` scratch buffer stores `1/√|S_a|²` for all fields, reused across timesteps to avoid repeated sqrt calls. The computation is parallelized via OpenMP reduction over edges.

### 2.4 Lyapunov Function

The Lyapunov function L is a scalar diagnostic that decomposes into non-interacting and interacting parts:

```
L = L₀ + Lc
```

**Non-interacting** (depends on each field's magnitude and pump):

```
L₀ = −½·Σ_a [ ½·h_a·Σ_k (Re(S_{ak})² − Im(S_{ak})²) − g·|S_a|² − ½β·|S_a|⁴ ]
```

**Interacting** (depends on coupling):

```
Lc = −½·Σ_{a,b,k} [ Jr_{ab}·(Re_a·Re_b + Im_a·Im_b) + Ji_{ab}·(Im_a·Re_b − Re_a·Im_b) ]
```

The sign of dL/dt (estimated as `(L(t) − L(t−dt))/dt`) indicates whether the system is approaching a stable configuration (dL/dt < 0) or moving away from one (dL/dt > 0). 
Note: L is a proper Lyapunov function (dL/dt ≤ 0 guaranteed along trajectories) only in the absence of equalization. When equalization is active, it acts as an external drive that can make dL/dt temporarily positive.

**Implementation notes:**
- L0 is computed in a merged field loop alongside the amplitude variance DS², with OpenMP reduction
- Lc is computed in a merged edge loop alongside H (H whenever `compute_H=1`; Lc only additionally when `compute_full` is true)
- Both use the sparse edge list, not the dense J arrays

### 2.5 Pump Threshold

The **pump threshold** `hth` is the smallest pump amplitude h at which the linearized system becomes unstable (max eigenvalue > 0). The linearized matrix is a 2N×2N symmetric matrix combining J, h, and g:

```
Lmat = [ ½(½h − g)·δ_{ij} + ½Jr_{ij}          −½Ji_{ij}       ]
       [          −½Ji_{ij}            −½(½h + g)·δ_{ij} + ½Jr_{ij} ]
```

This matrix acts on the 2N-dimensional vector `(Re(S_0),...,Re(S_{N-1}), Im(S_0),...,Im(S_{N-1}))` linearized around the origin S = 0. The code scans h from 0 upward in steps of 0.00001 until `max_eigenvalue(Lmat) > 0`, up to a maximum of h = 5. The threshold is used to auto-set the pump via `hm = pth × hth` when `pth ≥ 0`.

**Heap usage:** The 2N×2N matrix is heap-allocated via `calloc`, not a stack VLA. This supports N up to at least 1000 (32 MB allocation for N=1000). At N=1000, the GSL eigenvalue computation takes seconds to minutes depending on hardware.

### 2.6 Why h = 0 for Complex J

When the coupling matrix J is **complex** (`type_coup = 2` or `type_coup = 1`), the pump threshold prescription and the dynamics behave differently than for real J:

- The pump acts on **S\*** (the complex conjugate of the field), coupling to a Pauli σ_x matrix in the Re/Im representation
- The imaginary part of J couples to a **different** Pauli matrix (σ_y) in the same representation
- These two matrices do **not** commute — meaning there is no common eigenbasis that diagonalizes both the pump and the coupling simultaneously
- Consequently, the driven-dissipative dynamics at h > 0 do **not** effectively minimize the Hamiltonian when J is complex

This is why all production benchmarks use `h = 0` (by setting `pth = -1` and `hm = 0` in the config) for complex coupling matrices. When J is real (`type_coup = 0`), the imaginary parts of the fields are damped out and the pump commutes with the coupling, so the dynamics does minimize the Hamiltonian effectively even at h > 0. The system still evolves via the dissipative dynamics (g, β, equalization) and can find ground states through the interplay of coupling and dissipation.

#### Time gates

Three time-dependent switches control when each term activates:

| Gate | Activation time | Effect |
|------|----------------|--------|
| `t_coup` | Coupling on | f_J starts contributing. `-1` = never on. |
| `t_eq` | Equalization on | f_eq starts contributing. `-1` = never on. |
| `t_pump` | Pump modification | Permanent offset Dh added to all h_a. `-1` = never. |
| `t_pump2` | Second pump modification | Offset Dh2 added to all h_a once at t = t_pump2 (independent of t_pump). `-1` = never. |

Time gates allow staged activation of different physical processes. Common strategies:
- `t_eq < t_coup`: Let equalization stabilize norms before coupling turns on
- `t_eq = t_coup`: Start coupling and equalization simultaneously (standard choice)
- `t_eq > t_coup`: Let coupling evolve alone first, then enforce equalization
- `t_pump = -1`: Never apply the pump offset
- `t_pump2 = -1`: Never apply the second pump offset (default). Set both `t_pump`/`Dh` and `t_pump2`/`Dh2` to build a second desired-pump cycle (e.g. the R=3 re-anneal recipe). `Dh2` defaults to 0.

---

## 3. Numerical Integration

### 3.1 RK2 Midpoint Method

The time-stepping uses a second-order Runge-Kutta method (midpoint rule):

```
MIDPOINT:   S̃ = S + ½·dt·f(S)     (half-step prediction)
UPDATE:     S′ = S + dt·f(S̃)       (full step from midpoint)
```

Each RK2 step has two phases, each requiring one J×S matrix-vector product:

**Midpoint phase:**
1. Compute `J·S` from current field state (→ `JS_re`, `JS_im`)
2. For each field a (parallel loop):
   - Compute non-interacting coefficients f₀ using precomputed cos/sin
   - Add coupling contribution `½·JS` if `coup_control`
   - Add equalization if `eq_control`
   - Store midpoint state in `DS_re`, `DS_im` (half-step)
3. Compute midpoint norms `DS2`

**Update phase:**
1. Compute `J·DS` from midpoint state (→ `JS_re`, `JS_im`)
2. For each field a (parallel loop):
   - Compute full-step non-interacting coefficients using midpoint norms
   - Apply full dt to all contributions
   - Update `S_re`, `S_im` in place
   - Update `Y_re`, `Y_im` in place
3. Compute final norms `S2`

### 3.2 Stability and dt Selection

The relevant time scales are set by the pump h, dissipation g, nonlinearity β·|S|², coupling spectral radius |λ_J|, and equalization rate deq. RK2 stability requires dt to resolve the fastest of these:

```
dt ≲ C / max(h, g, β·|S|², |λ_J|, deq)
```

where C ≈ 1–2 for RK2 (the stability region for a real eigenvalue λ extends to dt = 2/|λ|). In practice, C ≈ 0.5–1 provides a safety margin for the nonlinear terms.

The step size is **fixed** — there is no adaptive stepping. If the simulation becomes numerically unstable, reduce dt.

### 3.3 Precomputation Strategy

Three cost-saving measures are applied per timestep:

1. **Trigonometric precomputation:** `cos(φ_a)` and `sin(φ_a)` are computed once at startup (via `precompute_trig()`), since `phi` is constant throughout the run, then reused for all D components of each field in both RK2 phases. This avoids 4·N redundant trig calls per step.

2. **J×S product caching:** The coupling product is computed once per RK2 phase via `compute_coupling_product_flat()` and stored in `JS_re`/`JS_im`. All D components reuse the same J×S result. Without this caching, each field component would need its own O(N²) coupling sum.

3. **Scratch buffer reuse:** The `inv_sqrtS2` array is allocated once and reused every timestep for normalization to hyperspin in `Hamiltonian()` and `magnetization()`. This avoids `malloc/free` overhead at every output step.

---

## 4. Code Architecture

### 4.1 File Roles

| File | Lines | Role |
|------|-------|------|
| `src/Complex_HSM.h` | ~1808 | Header + all function implementations: data structures, coupling matrix generation (random, lattice, file), sparse CSR/edge-list construction, dynamics (RK2), observables (Hamiltonian, Lyapunov, magnetization), eigenvalue computation, output I/O, hash helpers |
| `src/Complex_HSM.c` | ~302 | Main program entry point: argument parsing, initialization branching (sparse-first, lattice mode, eigenvalue mode), main time-stepping loop, stop condition checking, cleanup |
| `Complex_HSM.ini` | — | Configuration file (see §7) |

All function implementations are in `Complex_HSM.h` (included once by `Complex_HSM.c`). There is no separate `.c` for the library functions — the header is both the declaration and the definition.

### 4.2 Main Loop Flow

```
1. parse_config(argv[3])         ← read Complex_HSM.ini (all parameters)
2. Override N = NL^d             ← if new_coupling == 2 (lattice mode)
3. allocate_system(N, D, &s)     ← allocate flat 1D arrays (calloc)
4. initialize_fields(&s, Amax, initial_type)   ← random field/equalizer initial values via rand()
5. compute_hyperspin_norm(&s)    ← compute |S_a|² for all fields
6. Branch on new_coupling:
   ├── 2 (lattice):
   │     if compute_eigenvalues: alloc dense J
   │     initialize_J_Hermitian_Lattice()
   │       → sparse built directly, writes Data_Matrix_sparse.dat
   │       → if dense: also writes Data_Matrix.dat
   ├── 1 (random):
   │     initialize_J_Hermitian_Binary_Sparse()
   │       → rand_r 3-pass: count → fill CSR → write sparse file
   │     if compute_eigenvalues: alloc dense J + fill_dense_from_sparse()
   │       → write Data_Matrix.dat
   └── 0 (file):
         try Data_Matrix_sparse.dat (sparse reader)
         → fallback to Data_Matrix.dat (dense reader, builds sparse)
         if compute_eigenvalues: alloc dense J + fill_dense_from_sparse()
           → write Data_Matrix.dat
7. if output_dir set:
     compute FNV-1a hash of ini + seed + matrix
     create results/run_<hash>/
     copy ini + matrix file(s) into folder
8. if pth >= 0: h_threshold()   ← compute pump threshold (needs dense J)
9. initialize_pump()             ← set h_a, φ_a
10. precompute_trig()            ← cos/sin for all φ_a
11. if compute_eigenvalues:
      Linearized_Max_Eigenvalue() ← Jmax, Lmax
12. initialize_pump(hmod, Dh)    ← prepare pump modification offset
13. init_output_files()          ← open files with buffered I/O
14. Main loop (t = 0 to tmax):
      ├── Check time gates (coup_control, eq_control, pump_control)
      ├── compute_observables(): H only on save / E-target / stability-check / final
      │       steps (compute_H gate); full set (L0, Lc, L, dL, m, DS²) on save_steps;
      │       skipped entirely on idle steps
      ├── save_output_files()     ← every save_steps steps
      ├── flush_output_files()    ← every flush_interval steps
      ├── RK2_step()              ← advance all fields+equalizers by dt
      └── Check stop conditions (target energy / energy stable / tmax)
15. save_final_config()          ← final state to Data_Final.dat
16. write_metadata()             ← if output_dir enabled
17. close_output_files()         ← fclose all handles
18. free_system(&s)              ← free all memory
```

### 4.3 Data Structures

**`system_state`** holds all simulation data in **flat contiguous 1D arrays** (not pointer chains) for cache efficiency:

| Array | Size | Description |
|-------|------|-------------|
| `S_re`, `S_im` | N × D | Field components (real/imag) |
| `S2` | N | Squared magnitude per field |
| `DS_re`, `DS_im` | N × D | Midpoint values (RK2 half-step) |
| `DS2` | N | Midpoint squared magnitudes |
| `Y_re`, `Y_im` | N × 2 | Equalizer variables |
| `DY_re`, `DY_im` | N × 2 | Midpoint equalizer values |
| `J_re`, `J_im` | N × N | Dense coupling matrix — only allocated if `compute_eigenvalues=1` |
| `h` | N | Pump amplitude per field |
| `hmod` | N | Pump modification offset (applied at t_pump) |
| `phi` | N | Pump phase per field |
| `JS_re`, `JS_im` | N × D | Precomputed J×S product |
| `cos_phi`, `sin_phi` | N | Precomputed trig values |
| `inv_sqrtS2` | N | Scratch buffer for normalization |
| `csr_row_ptr` | N+1 | CSR row pointers (int) |
| `csr_col_idx` | nnz | CSR column indices (int) |
| `csr_Jr`, `csr_Ji` | nnz | CSR coupling values (double) |
| `edge_a`, `edge_b` | num_edges | Undirected edge list indices (int) |
| `edge_Jr`, `edge_Ji` | num_edges | Undirected edge list values (double) |
| `J_allocated` | 1 | Flag: was dense J allocated? |
| `num_edges` | 1 | Number of undirected edges |
| `startup_heap_B` | 1 | Heap snapshot after coupling generation (memory benchmark) |
| `steady_heap_B` | 1 | Heap snapshot after I/O buffer allocation (memory benchmark) |

**`io_context`** holds all open `FILE*` handles:

| Field | File | Used by `save_dyn` |
|-------|------|--------------------|
| `f_fields` | `Data_Fields.dat` | 1 |
| `f_equalizers` | `Data_Equalizers.dat` | 1 |
| `f_hamilton` | `Data_Hamilton.dat` | 1, 2 |
| `f_lyapunov` | `Data_Lyapunov.dat` | 1, 2 |
| `f_h_ly_m_ds2` | `Data_H_Ly_m_DS2.dat` | 3 |
| `f_final` | `Data_Final.dat` | all |

### 4.4 Index Macros

All arrays are 1D. Macros provide readable access with implicit offset computation:

```c
// Field component: field a, component j
#define S_RE(s, a, j)  (s)->S_re[(a)*(s)->D + (j)]
#define S_IM(s, a, j)  (s)->S_im[(a)*(s)->D + (j)]
#define S2(s, a)       (s)->S2[a]
#define DS_RE(s, a, j) (s)->DS_re[(a)*(s)->D + (j)]
#define DS_IM(s, a, j) (s)->DS_im[(a)*(s)->D + (j)]
#define DS2(s, a)      (s)->DS2[a]

// Equalizer: field a, k = 0 or 1
#define Y_RE(s, a, k)  (s)->Y_re[(a)*2 + (k)]
#define Y_IM(s, a, k)  (s)->Y_im[(a)*2 + (k)]
#define DY_RE(s, a, k) (s)->DY_re[(a)*2 + (k)]
#define DY_IM(s, a, k) (s)->DY_im[(a)*2 + (k)]

// Coupling matrix (dense): field a, field b
#define J_RE(s, a, b)  (s)->J_re[(a)*(s)->N + (b)]
#define J_IM(s, a, b)  (s)->J_im[(a)*(s)->N + (b)]

// Precomputed coupling product
#define JS_RE(s, a, j) (s)->JS_re[(a)*(s)->D + (j)]
#define JS_IM(s, a, j) (s)->JS_im[(a)*(s)->D + (j)]
```

The striding is `D` per field for field arrays and `2` per field for equalizer arrays. The dense coupling matrix uses `N` stride per row.

### 4.5 Memory Layout (Contiguous Arrays)

All arrays are allocated as single `calloc` blocks, not arrays of pointers. This gives:
- **Better cache locality:** Sequential field access corresponds to sequential memory access
- **No indirection overhead:** `S_RE(s, a, j)` is a single offset computation, not a pointer dereference
- **Simpler memory management:** One `free()` per array, no nested deallocation
- **Zeroed memory:** `calloc` ensures all arrays start at zero (important for `J_re`, `J_im` which build from sparse structures incrementally)

---

## 5. Sparse-First Architecture

### 5.1 Motivation

The coupling matrix J is N×N, requiring O(N²) memory and computation when stored densely. For **sparse connectivity** (e.g., lattices where each hyperspin connects to ~2d neighbors, or random graphs with mean degree k ≪ N), O(N²) operations are wasteful:

- A 2D lattice of N=10⁶ hyperspins has ~4·10⁶ non-zero couplings, but a dense N² storage would need 10¹² entries
- Even a moderately sparse random graph (mean degree k = 10) at N=10⁶ has ~10⁷ non-zero entries — still trivial compared to dense storage

The sparse-first architecture eliminates all O(N²) allocations and operations except when eigenvalue computation explicitly requires the dense matrix.

### 5.2 Two Sparse Representations

Two complementary representations are built once at startup:

#### CSR (Compressed Sparse Row)

Used by `compute_coupling_product_flat()` for the J×S matrix-vector product.

```
csr_row_ptr[0] = 0
For each field a:
  csr_row_ptr[a+1] = csr_row_ptr[a] + degree(a)
  For each neighbor b of a:
    csr_col_idx[k] = b
    csr_Jr[k] = Jr_{ab}
    csr_Ji[k] = Ji_{ab}
    k++
```

The CSR stores **directed** (a,b) pairs: for each undirected edge, both (a,b) and (b,a) are stored with their respective J values (Jr + i·Ji for upper triangle, Jr − i·Ji for lower). The coupling product loops:

```c
for (k = row_ptr[a]; k < row_ptr[a+1]; k++) {
    int b = col_idx[k];
    out_re[a*D + j] += Jr[k] * src_re[b*D + j] - Ji[k] * src_im[b*D + j];
    out_im[a*D + j] += Jr[k] * src_im[b*D + j] + Ji[k] * src_re[b*D + j];
}
```

#### Undirected Edge List

Used by `compute_observables()` for H and Lc.

```
For each undirected edge (a < b):
  edge_a[e] = a
  edge_b[e] = b
  edge_Jr[e] = Jr_{ab}
  edge_Ji[e] = Ji_{ab}
```

Since the (a,b) and (b,a) contributions to both Lc and H are identical (Hermitian symmetry), the edge list processes each pair once with a ×2 factor. This halves the inner-loop iterations compared to a full double loop.

### 5.3 All Paths Build Sparse First

Every coupling matrix initialization path constructs the sparse structures **directly**, without a dense intermediate or a `build_sparse_coupling()` conversion step:

#### Path 1: Random Graph (`new_coupling = 1`)

`initialize_J_Hermitian_Binary_Sparse()` uses a rewindable local RNG (`rand_r`) with three reseeded passes:

```
Pass 1 (count):  seed = NS; for each pair (i,j) with j<i:
                   exists = rand_r(&seed) < prob_coup
                   if exists: row_count[i]++, row_count[j]++, nnz += 2
                   (consume sign_re + sign_im random values but discard — just advancing RNG)
                 → nnz known, allocate CSR + edge list

Pass 2 (fill):   seed = NS; for each pair (i,j) with j<i:
                   exists = rand_r(&seed) < prob_coup
                   if exists: sign_re = rand_r(&seed) < prob_pm_re
                              sign_im = rand_r(&seed) < prob_pm_im
                   → same decisions as pass 1 because same seed
                   → fill CSR + edge list

(Pass 3 is implicit via write_sparse_matrix_file which reads from the edge list)
```

This eliminates the `pair_data` temporary array (0.5·N² bytes) that earlier versions used to store one bit per upper-triangle pair. No O(N²) heap allocation occurs at any point.

#### Path 2: Lattice (`new_coupling = 2`)

`initialize_J_Hermitian_Lattice()` uses a single forward pass with an `edge_decisions` array:

```
For each hyperspin a:
  index_to_coord(a, NL, d, coord)
  For each dimension dim:
    Forward neighbor b: coord[dim]+1 (with PBC wrap or OBC check)
    if b > a: exists = rand() < prob_coup, sign_re = rand() < prob_pm_re, sign_im = rand() < prob_pm_im
    Store {b, exists, sign_re, sign_im} in edge_decisions[]

    Backward neighbor b: coord[dim]-1 (with PBC wrap or OBC check)
    if b > a: exists = rand() < prob_coup, sign_re = rand() < prob_pm_re, sign_im = rand() < prob_pm_im
    Store {b, exists, sign_re, sign_im} in edge_decisions[]

→ row_count built during decision generation
→ Allocate CSR + edge list
→ Second pass over edge_decisions[] fills CSR + edge list
→ Free edge_decisions[]
```

The `edge_decisions` array is `24·d·N` bytes (2·d·N entries × 12 bytes each: three ints per entry). For N=10⁶, d=3, this is 72 MB — a temporary that is freed immediately after construction.

#### Path 3: File Input (`new_coupling = 0`)

Two sub-paths with automatic detection:

1. **Sparse file (`Data_Matrix_sparse.dat`):** `initialize_J_Sparse_From_Sparse_File()`
   - Read `num_edges` from first line
   - Two-pass: count per-row degree → allocate → fill CSR + edge list
   - O(nnz) time, O(nnz) memory

2. **Dense file (`Data_Matrix.dat`):** `initialize_J_Sparse_From_File()`
   - Legacy N×N grid format (space-separated "Re Im" per cell)
   - Two-pass file read: count non-zero off-diagonals → allocate → fill
   - O(N²) file read time (unavoidable — must scan all entries to find non-zero ones)
   - O(nnz) memory for sparse structures (the file reading is O(N²), but storage is O(nnz))

#### Path 4: Dense J Fills (for `compute_eigenvalues = 1`)

When eigenvalue computation is needed (`compute_eigenvalues=1`), dense `J_re`/`J_im` are allocated (16·N² bytes) after sparse construction and filled via `fill_dense_from_sparse()`:

```c
memset(J_re, 0, N*N*sizeof(double));
memset(J_im, 0, N*N*sizeof(double));
for (e = 0; e < num_edges; e++) {
    a = edge_a[e], b = edge_b[e];
    J_RE(s, a, b) = edge_Jr[e];     // Jr + i·Ji
    J_IM(s, a, b) = edge_Ji[e];
    J_RE(s, b, a) = edge_Jr[e];     // Jr − i·Ji  (Hermitian conjugate)
    J_IM(s, b, a) = -edge_Ji[e];
}
```

This is O(nnz) — fast. The dense J is only used for:
- `h_threshold()` (scanning h for pump threshold)
- `Linearized_Max_Eigenvalue()` (computing Jmax, Lmax)

The dynamics never touches the dense J arrays.

### 5.4 Memory Comparison

Memory at N=10⁴, prob_coup=0.7 (random graph):

| Component | `compute_eigenvalues=1` (dense) | `compute_eigenvalues=0` (sparse) |
|-----------|:-------------------------------:|:--------------------------------:|
| Dense J_re + J_im | 2 × 10⁸ × 8 = 1.6 GB | 0 B |
| Eigenvalue scratch (4·N²) | 4 × 10⁸ × 8 = 3.2 GB | 0 B |
| CSR + edge list (32·nnz) | ~2.2 GB | ~2.2 GB |
| **Total** | **~7.0 GB** | **~2.2 GB** |

Memory at N=10⁶, mean degree k=10 (sparse random graph, compute_eigenvalues=0):

| Component | Size |
|-----------|:----:|
| Core arrays (D=2) | 48·N·D + 128·N = 224·N ≈ 224 MB |
| Sparse structures (4·N + 4 + 32·nnz, nnz=k·N) | 4·N + 320·N ≈ 324 MB |
| I/O buffers | ~512 KB |
| **Total** | **~548 MB** |

### 5.5 When to Use compute_eigenvalues

| System size | Recommendation | Reason |
|-------------|---------------|--------|
| N < 500 | `compute_eigenvalues = 1` | Dense J is small (16·N² < 4 MB), eigenvalues are fast, diagnostics valuable |
| N = 500–2000 | Either | Dense J uses 4–64 MB — acceptable. Eigenvalue computation takes seconds to minutes |
| N > 2000 | `compute_eigenvalues = 0` | Dense J exceeds 64 MB; eigenvalue scratch adds 32·N²; diagonalization becomes slow (hours at N=10⁴) |
| N > 10⁵ | Must be `compute_eigenvalues = 0` | Dense J alone would be many GB/TB |

**Important:** When `compute_eigenvalues = 0`, `pth` must be `-1` (no auto-threshold). The program errors out if `pth >= 0` and `compute_eigenvalues = 0`.

### 5.6 Determinism

Within v2.6, same seed + same config → same coupling matrix and same dynamics:
- **Random graph (`new_coupling=1`):** The `rand_r` 3-pass generator is fully deterministic. Same NS → same sequence of pair decisions.
- **Lattice (`new_coupling=2`):** Uses `rand()` for edge decisions, seeded via `srand(NS)` in main(). Same NS → same lattice.
- **File input (`new_coupling=0`):** Matrix file is read verbatim — identical input → identical sparse structures.
- **`initial_type` RNG invariant:** `initialize_fields` always draws both the real and imaginary components (unconditional `rand()` calls) and then zeroes the unused component. Never skip the second draw when implementing constrained initial conditions — skipping would shift the `rand()` stream and silently change the lattice coupling matrix for the same NS. Because the draws are unconditional, `initial_type=2` reproduces the v2.5 dynamics byte-for-byte, and the lattice matrix is identical across all `initial_type` values.

---



## 6. Coupling Matrix Generation

### 6.1 Random Binary Coupling (`new_coupling = 1`)

#### Algorithm

The function `initialize_J_Hermitian_Binary_Sparse()` generates an N×N Hermitian random coupling matrix using a rewindable local RNG (`rand_r`) with three reseeded passes.

**Normalization:**

The coupling magnitude depends on N, `prob_coup`, and `type_coup`:

| `type_coup` | Real magnitude (dJR) | Imag magnitude (dJI) |
|:-----------:|:--------------------:|:--------------------:|
| 0 (real only) | `1 / (N · prob_coup)` | 0 |
| 1 (imag only) | 0 | `1 / (N · prob_coup)` |
| 2 (both Re+Im) | `1 / (√2 · N · prob_coup)` | `1 / (√2 · N · prob_coup)` |

For `type_coup = 2`, the √2 factor splits the total magnitude equally between Re and Im while keeping the total RMS coupling the same as type_coup = 0 or 1.

**Hermitian structure:**

For each pair (i, j) with j < i:
- Draw `exists` ~ Bernoulli(prob_coup). If 0: both J(i,j) and J(j,i) are zero.
- If exists: draw `sign_re` ~ Bernoulli(prob_pm_re), `sign_im` ~ Bernoulli(prob_pm_im). sign = 0 → negative, sign = 1 → positive.
- J(i,j) = dJR·coup_sign + i·dJI·coup_sign   (lower triangle, i > j)
- J(j,i) = dJR·coup_sign − i·dJI·coup_sign   (upper triangle, j < i, Hermitian conjugate)

This ensures Hermiticity: J_re is symmetric, J_im is antisymmetric.

**CSR storage convention:**

For edge (a,b) with a < b:
- CSR row a → col b: stores J(a,b) = dJR·coup_sign − i·dJI·coup_sign  (upper triangle)
- CSR row b → col a: stores J(b,a) = dJR·coup_sign + i·dJI·coup_sign  (lower triangle)
- Edge list: stores (a, b, Jr, Ji) where Jr = dJR·coup_sign, Ji = −dJI·coup_sign

The `-J2` variable in the source code:
```c
double J2 = -dJI * coup_sign;  // J_IM(a,b) for a < b
```
Then:
```c
// CSR row a, col b (upper triangle): J(a,b) = J1 + i*J2 = dJR·coup_sign - i·dJI·coup_sign
s->csr_Ji[ka] = J2;   // = -dJI·coup_sign = J_IM(a,b)

// CSR row b, col a (lower triangle): J(b,a) = J1 + i*(-J2) = dJR·coup_sign + i·dJI·coup_sign
s->csr_Ji[kb] = -J2;  // = +dJI·coup_sign = J_IM(b,a)
```

### 6.2 Hypercubic Lattice (`new_coupling = 2`)

#### Coordinate Convention

The d-dimensional hypercube has NL sites per dimension in C row-major order (coord[0] varies fastest):

```c
// index_to_coord: extract base-NL digits, LSB to MSB
// coord[0] = least significant digit (fastest-varying)
void index_to_coord(int idx, int NL, int d, int *coord) {
    for (int dim = 0; dim < d; dim++) {
        coord[dim] = idx % NL;
        idx /= NL;
    }
}

// coord_to_index: reconstruct index from coordinate
int coord_to_index(const int *coord, int NL, int d) {
    int idx = 0;
    int stride = 1;
    for (int dim = 0; dim < d; dim++) {
        idx += coord[dim] * stride;
        stride *= NL;
    }
    return idx;
}
```

For d=2, NL=3:
```
Index   Coord     Visual
  0    (0,0)     0  1  2    (y=0 row)
  1    (1,0)     3  4  5    (y=1 row)
  2    (2,0)     6  7  8    (y=2 row)
  3    (0,1)
  4    (1,1)     coord[0] = x (fastest-varying)
  5    (2,1)     coord[1] = y (slower)
  6    (0,2)
  7    (1,2)
  8    (2,2)
```

For d=3, coord[2] is the z-coordinate (slowest), and so on.

#### Edge Generation Algorithm

Each hyperspin has up to `2·d` potential neighbors (forward and backward in each dimension). The algorithm generates an edge decision for each potential neighbor, but only stores edges where b > a (upper-triangle filtering to avoid duplicates).

**Dual-direction checking:** For each dimension dim, both forward (coord[dim]+1) and backward (coord[dim]-1) neighbors are examined. This is essential for PBC: a hyperspin at position 0 in a given dimension has both a forward neighbor (1) and a backward neighbor (NL-1) — both may have higher or lower index depending on the dimension. Without dual-direction checking, PBC wrap-around edges would be missed when they happen to have b < a.

**Boundary handling (OBC):** When a hyperspin is at coord[dim] = 0 (backward) or coord[dim] = NL-1 (forward) and `lattice_bc = 0`, that direction has no neighbor. The direction is marked invalid and no edge decision is generated. Placeholder entries with `exists = 0` are inserted into the decision array to maintain lockstep between passes.

**Independent edge decisions per dimension:** Each potential edge has independent random existence and sign. At `prob_coup = 1`, all 2·d edges per hyperspin exist. At `prob_coup < 1`, some fraction are missing, creating a random bond percolation on the lattice.

#### Normalization

The coupling strength matches the energy scale of the random binary coupling. The degree is `2·d` (maximum coordination number in PBC):

| type_coup | Jr magnitude | Ji magnitude |
|:---------:|:------------:|:------------:|
| 0 (real only) | `1/(2·d·prob_coup)` | 0 |
| 1 (imag only) | 0 | `1/(2·d·prob_coup)` |
| 2 (both Re+Im) | `1/(√2·2·d·prob_coup)` | `1/(√2·2·d·prob_coup)` |

**Hermitian structure:**

For each undirected edge (a, b) with coupling magnitude J1 = dJR·sign and J2 = dJI·sign:

| Matrix position | Real part | Imaginary part |
|----------------|-----------|----------------|
| J(a, b) [a < b] | J1 | −J2 |
| J(b, a) [b > a] | J1 | +J2 |

This matches the convention used by the random binary generator.

#### Topology Verification (2D 3×3, N=9, prob_coup=1)

**PBC:** Each hyperspin connects to exactly 4 neighbors (2·d = 4). All 18 undirected edges exist.

```
Hyperspin 0 at (0,0): forward-x=1(1,0), back-x=2(2,0), forward-y=3(0,1), back-y=6(0,2)
Hyperspin 1 at (1,0): connected to 0, 2, 4, 7
...
Total: 9 hyperspins × 4 neighbors / 2 = 18 unique edges
```

**OBC:** Corner hyperspins have 2 neighbors, edge (non-corner) hyperspins have 3, interior hyperspin 4=(1,1) has 4.

```
Hyperspin 0 at (0,0): neighbors = {1=(1,0), 3=(0,1)}   (no backward-x or backward-y)
Hyperspin 4 at (1,1): neighbors = {1=(1,0), 3=(0,1), 5=(2,1), 7=(1,2)}
Total: 12 undirected edges
```

### 6.3 File Input (`new_coupling = 0`)

The program auto-detects which file format to read:

1. **Try `Data_Matrix_sparse.dat` first** (sparse edge-list format) → `initialize_J_Sparse_From_Sparse_File()`
2. **Fall back to `Data_Matrix.dat`** (dense N×N grid) → `initialize_J_Sparse_From_File()`

Both functions build the CSR + edge list directly from the file data. The sparse reader is O(nnz), while the dense reader is O(N²) (must scan all entries to find non-zero ones).

### 6.4 Sparse File I/O

#### File Format: `Data_Matrix_sparse.dat`

```
num_edges
a1  b1  Jr1  Ji1
a2  b2  Jr2  Ji2
...
```

- First line: total number of undirected edges (integer)
- Each subsequent line: one undirected edge (a < b), with J(a,b) = Jr + i·Ji
- Hermitian symmetry is implicit: J(b,a) = Jr − i·Ji (derived, not stored)
- No zero entries — the file is O(nnz), not O(N²)

#### I/O Rules Table

| Generation mode | Sparse file written? | Dense file written? |
|:---------------:|:--------------------:|:-------------------:|
| `new_coupling=1`, `compute_eigenvalues=0` | ✅ Yes (O(nnz)) | ❌ No |
| `new_coupling=1`, `compute_eigenvalues=1` | ✅ Yes (O(nnz)) | ✅ Yes (O(N²), from dense J) |
| `new_coupling=2`, `compute_eigenvalues=0` | ✅ Yes (O(nnz)) | ❌ No |
| `new_coupling=2`, `compute_eigenvalues=1` | ✅ Yes (O(nnz)) | ✅ Yes (O(N²), from dense J) |

#### Format Converter Tools

The `scripts/` directory contains two standalone C utilities for converting between formats:
- `matrix_dense_to_sparse.x`: converts `Data_Matrix.dat` → `Data_Matrix_sparse.dat`
- `matrix_sparse_to_dense.x`: converts `Data_Matrix_sparse.dat` → `Data_Matrix.dat`

These are standalone programs (not part of Complex_HSM.x), useful for preparing matrix files for the sparse-only code path.

### 6.5 Data_Matrix.dat Format (Dense)

The legacy dense format is still used for backward compatibility:

```
\n
Re(0,0)  Im(0,0)    Re(0,1)  Im(0,1)    ...  Re(0,N-1)  Im(0,N-1)
Re(1,0)  Im(1,0)    Re(1,1)  Im(1,1)    ...  Re(1,N-1)  Im(1,N-1)
...
Re(N-1,0)  Im(N-1,0)  ...        Re(N-1,N-1)  Im(N-1,N-1)
```

Each cell is two space-separated doubles (Re, Im). Rows are newline-separated. The file is O(N²) — ~40·N² bytes for N×N pairs of "%0.15f" doubles.

---

## 7. Configuration Reference

All parameters are set in `Complex_HSM.ini` using `key = value` syntax. Lines starting with `#` are comments.

> **Note:** The example values below are illustrative and differ from the
> shipped `Complex_HSM.ini`, which uses production-oriented settings
> (e.g. `new_coupling = 1`, `prob_pm_re = 0.5`, `prob_pm_im = 0.5`, `coup_phase = 0`, `type_coup = 2`,
> `tmax = 1000000`, `save_steps = 1000`, `Amax = 0.1`). Every key, default,
> and range documented here is authoritative; only the example numbers vary.

### 7.1 System

```ini
N   = 100   # Number of hyperspins (integer, ≥ 1)
D   = 2     # Complex components per hyperspin (integer, ≥ 1)
```

`D` is the hyperspin dimension, NOT the lattice spatial dimension. A d-dimensional lattice of N = NL^d hyperspins can have any D.

### 7.2 Initial Conditions

```ini
initial_type = 2 # Initial condition type: 0 = real (Im ≡ 0), 1 = pure imaginary
                 # (Re ≡ 0), 2 = general complex. Applies to both the S fields
                 # (N×D) and the Y0/Y1 equalizers (N×2).
                 # Values outside 0-2 are rejected: error + exit 1.

Amax = 1.0   # Max initial amplitude. Each field component and equalizer
             # initialized uniformly in [-Amax, +Amax].

phim = 0.0   # Average pump phase (radians)
dphi = 0.0   # Max random phase deviation. φ_a = phim + uniform[-dphi, dphi].

hm   = 0.0   # Average pump amplitude. Can be overridden by auto-threshold
             # when pth ≥ 0 on the command line.
dh   = 0.0   # Max random pump deviation. h_a = hm + uniform[-dh, dh].

Dh   = 0.0   # Permanent pump offset applied at t = t_pump. All h_a
             # increase by Dh when the pump gate triggers.
Dh2  = 0.0   # Second pump offset applied at t = t_pump2. All h_a
             # increase by Dh2 when the second pump gate triggers
             # (independent of t_pump). Default 0 = disabled.
```

**Note:** For complex couplings (`type_coup = 2`), the pump and coupling terms do not commute. Use `hm = 0` (pth = −1 on the command line) unless you specifically need pump-driven dynamics. This is the standard protocol for ground-state finding.

**Note (initial_type purity):** The initial Re/Im constraint in modes 0 and 1 is preserved by the dynamics **exactly** only when the coupling is real-only (`type_coup = 0`) and the pump cross term `½·h·sin(φ)` vanishes — i.e. `phim = 0` and `dphi = 0` (the pump amplitude `h` itself is irrelevant to this term). With `type_coup = 2`, or any nonzero pump phase, the dynamics immediately generates the opposite component. The stock ini (`phim = dphi = 0`, `hm = 0`) satisfies the constraint, so mode 0/1 runs stay exactly real/imaginary throughout.

### 7.3 Dynamics

```ini
gm   = 0.0   # Intrinsic loss rate (g in equations). Positive = damping.
             # Controls how quickly the system decays toward the ground state.
beta = 0.1   # Nonlinear saturation coefficient. Suppresses large-amplitude
             # |S|² terms in the dynamics. Prevents amplitude blow-up when
             # the origin is no longer an attractor.
deq  = 1.0   # Equalization coupling strength. Higher = faster equalization
             # of field norms across the ring topology.
```

### 7.4 Stop Conditions

Two independent stop conditions. The simulation stops when either is met, or when t = tmax (whichever comes first).

```ini
stop_at_E_target   = 0      # 1 = stop when H ≤ E_target × (1 − E_target_tol)
E_target           = -1.0   # Target energy value
E_target_tol       = 1e-10  # Tolerance: stop when H goes below E_target*(1-tol)

stop_when_stable   = 1      # 1 = stop when |H − H_prev| ≤ tolerance
                            #     for stable_window consecutive checks
allowed_H_change   = 1e-12  # Max allowed energy variation between checks
                            #   (absolute or relative, see relative_H_change)
steps_for_update   = 1      # Time steps between stability checks [default: 1]
stable_window      = 1000   # Consecutive stable checks needed to trigger stop
relative_H_change  = 0      # 0 = absolute check: |H−H_prev| ≤ allowed_H_change [default]
                            # 1 = relative check: |H−H_prev|/|H| < allowed_H_change
                            #     Relative mode is N-scalable; absolute mode may be
                            #     too strict for large N (use relative_H_change=1
                            #     for N > 10^4)
```

The stability check requires `t > t_coup` and `t > t_eq`. When these are set to positive values, the check is deferred until after both gates activate. When set to `-1` (never on), the condition `t > -1` is always true, so the stability check proceeds immediately — but note that coupling and equalization remain off, so the system evolves under pump + dissipation + nonlinearity only.

### 7.5 Timing

```ini
tmax   = 20000.0  # Maximum simulation time. The simulation will stop here
                  # if no other stop condition triggers first.
dt     = 0.01     # Integration time step. RK2 stability requires
                  # dt ≲ C / max(h, g, β·|S|², |λ_J|, deq) where C ≈ 1–2.
                  # Typical range: 0.001–0.1.
t_coup = 50.0     # Time when coupling term activates. -1 = never on.
t_eq   = 50.0     # Time when equalization activates. -1 = never on.
t_pump = -1       # Time when pump modification (Dh) is applied.
                  # -1 = never.
t_pump2 = -1      # Time when second pump modification (Dh2) is applied.
                  # -1 = never (default).
```

**Note:** Convergence time varies enormously — even at fixed N, t_final can span 3 orders of magnitude depending on the matrix instance and initial conditions. Set `tmax` generously; the stability stop condition will terminate early when the system converges.

### 7.6 Coupling Matrix

```ini
new_coupling = 2    # 0 = read from file, 1 = generate new random matrix,
                    # 2 = generate d-dimensional hypercubic lattice
prob_coup    = 1.0  # Probability that a given pair of fields is coupled (0–1)
prob_pm_re   = 0.7  # Probability that Re coupling is positive (vs negative) (0–1)
prob_pm_im   = 0.7  # Probability that Im coupling is positive (vs negative) (0–1)
coup_phase   = 0    # Global coupling rotation angle in radians (applied once after construction)
type_coup    = 0    # 0 = real only, 1 = imaginary only, 2 = both Re+Im

# Lattice-specific parameters (only used when new_coupling = 2):
lattice_d       = 2   # Number of spatial dimensions (default: 2)
lattice_NL      = 10  # Linear size per dimension (default: 10)
                      # N is overridden to NL^d when new_coupling = 2
lattice_bc      = 1   # Boundary conditions: 0 = open (OBC), 1 = periodic (PBC)
```

prob_coup controls the edge density:
- `prob_coup = 1.0`: All possible edges exist (fully connected for random, or all lattice bonds present)
- `prob_coup < 1.0`: Random edge percolation (both for random and lattice modes)

For lattice generation, prob_coup < 1 creates a **random bond percolation** on the lattice topology: each of the 2·d possible edges per hyperspin independently exists with probability prob_coup.

`prob_pm_re` and `prob_pm_im` control the sign distribution independently for Re and Im coupling components. Each sign is drawn as `sign = rand() < prob_pm_re` (or `prob_pm_im`) (0 → negative, 1 → positive), so `0.5` gives equal FM/AFM mix and `1.0` gives all-positive coupling.

`coup_phase` applies a global rotation `e^{i·coup_phase}` to every coupling element after construction. The rotation is applied once in `main()` for all matrix construction paths (random graph, lattice, file-loaded), and the rotated matrix is saved to the sparse file. Set to `0` for no rotation. Common values: `π/4 ≈ 0.7854` rotates a purely real coupling to purely imaginary.

Compute_eigenvalues controls whether dense J is allocated:

```ini
compute_eigenvalues = 1    # 0 = sparse-only mode (no dense J, no eigenvalues),
                           # 1 = allocate dense J, compute Jmax, Lmax, hth
```

When `compute_eigenvalues = 0`:
- Dense J is never allocated (saves 16·N² bytes)
- `h_threshold()` is skipped (pth must be -1)
- `Linearized_Max_Eigenvalue()` is skipped (Jmax/Lmax printed as "N/A")
- Only sparse file (`Data_Matrix_sparse.dat`) is written — no dense `Data_Matrix.dat`

**Note:** Convergence results can vary significantly between matrix instances with identical parameters (same prob_pm_re, prob_pm_im, prob_coup, type_coup). The specific sign/phase pattern matters more than the average bias. Always run multiple seeds to get reliable statistics.

### 7.7 Output

```ini
save_dyn    = 3       # Output mode (0, 1, 2, or 3 — see §8)
save_steps  = 1       # Write output every N steps (1 = every step); values < 1 are clamped to 1
buffer_size = 262144  # I/O buffer size in bytes (default 256 KB)
flush_interval = 1000 # Flush buffers to disk every N steps
```

### 7.8 Output Directory

```ini
# output_dir = 1       # 0 = files in cwd (default), 1 = save in results/run_<hash>/
#                      # any other string = used verbatim as a base output path
```

The boolean shorthand `0`/`1` is mapped in main(): `1` → `results`, `0` → current working directory. Any other string is used verbatim as the base output path (the `results/run_<hash>/` subfolders are then created under it).

When `output_dir = 1`, the program creates `results/run_<16-char-hex-fnv1a-hash>/` and writes all output there. The hash is computed from:
1. ini file contents
2. Random seed (NS)
3. Coupling matrix file content
4. pth command-line argument

This ensures **deterministic folder naming**: identical inputs always produce the same folder; different parameters get separate folders automatically.

Each run folder contains:
- `Complex_HSM.ini` — copy of the config file
- `Data_Matrix_sparse.dat` and/or `Data_Matrix.dat` — copies of coupling matrix files
- All simulation output files
- `metadata.txt` — run summary (timestamp, parameters, H_final, t_final, stop reason)

### 7.9 Parallel

```ini
num_threads = 1   # 1 = sequential, >1 = parallel with that many threads
```

No recompilation needed — OpenMP thread count is set at runtime via `omp_set_num_threads()`.

---

## 8. Output Files

### 8.1 save_dyn Mode Reference

| `save_dyn` | Files created |
|:----------:|---------------|
| 0 | `Data_Final.dat` — final hyperspin state only |
| 1 | `Data_Final.dat`, `Data_Fields.dat`, `Data_Equalizers.dat`, `Data_Hamilton.dat`, `Data_Lyapunov.dat` |
| 2 | `Data_Final.dat`, `Data_Hamilton.dat`, `Data_Lyapunov.dat` |
| 3 | `Data_Final.dat`, `Data_H_Ly_m_DS2.dat` — merged observables |

### 8.2 File Formats

#### `Data_Final.dat`

One line per run, space-separated:
```
NS  T  S1  Re(A11)  Im(A11)  Re(A12)  Im(A12)  ...  S2  Re(A21)  Im(A21)  ...
```

Header: `NS  T  S1  Re(A11)  Im(A11)  Re(A12)  Im(A12)  ...  S2  Re(A21)  Im(A21)  ...`

For each hyperspin i (1-indexed): the squared norm S_i, then D pairs of Re/Im for each component. Note the `S1`, `S2`, … labels are the per-hyperspin squared norms, not component labels. All values printed with 15 decimal places.

#### `Data_Hamilton.dat` (save_dyn 1, 2)

```
t    H
```

Time and Hamiltonian energy, one row per output step. `%0.15f  %0.15f` format.

#### `Data_Lyapunov.dat` (save_dyn 1, 2)

```
t    L    dL    L0    Lc
```

Time, total Lyapunov function, time derivative dL/dt, non-interacting part L0, interacting part Lc. `%0.15f %0.15f %0.15f %0.15f %0.15f` format.

#### `Data_Fields.dat` (save_dyn = 1)

One row per field per output time:
```
t  spin_id  |S|²  Re(A1)  Re(A2)  ...  Im(A1)  Im(A2)  ...
```

N rows per time point. All values with 15 decimal places.

#### `Data_Equalizers.dat` (save_dyn = 1)

```
t  Re(Y0)  Re(Y1)  Re(Y2)  Re(Y3)  ...  Im(Y0)  Im(Y1)  ...
```

One row per time point. Only **even-indexed** equalizers are output: the code iterates `a = 0, 2, 4, …` and writes each equalizer's two components `Y_RE(s, a, 0)` and `Y_RE(s, a, 1)` as a column pair, all Re first then all Im. Each equalizer Y_a = (Y_{a,0}, Y_{a,1}) connects the two hyperspin fields (a, a+1) in the equalization ring: Y_{a,0} absorbs positive norm differences and Y_{a,1} negative ones. The header labels each column pair with the indices `(a, a+1)` (e.g. `Re(Y0) Re(Y1)`, then `Re(Y2) Re(Y3)`, …) — so column 2 of each pair is the second *component* of equalizer a, not equalizer a+1, and the odd-indexed equalizers are never written.

#### `Data_H_Ly_m_DS2.dat` (save_dyn = 3)

```
t  H  Ly/N  <m>  <DS²>
```

Merged observable file: time, Hamiltonian, Lyapunov per field (L/N), average magnetization, amplitude variance (|Var|/N). Five columns, `%0.15f` format.

### 8.3 Output Throttling & Performance

- **`save_steps`**: Controls the frequency of both output writes and expensive observable computation. The `compute_H` gate decides whether any observable is computed at all: H (plus the `inv_sqrtS2` scratch and the edge pass) runs only on save steps, E-target runs (`stop_at_E_target=1`), the step before a stability check (via `(steps+1) % steps_for_update == 0`, so the check reads the same H canonical would compute), and the final step. The full set of observables (Lyapunov L, dL, magnetization m, amplitude variance DS²) is computed only when `steps % save_steps == 0` and data is written to file. On idle steps (none of the above, `save_dyn=0` benchmarks typical) the `compute_observables()` call is skipped entirely. `save_steps = 1` (default) computes everything every step. `save_steps = 10` reduces both computation and file I/O by 10×. At large N this saves ~40% of observable time.
- **`flush_interval`**: Buffer is flushed to disk every `flush_interval` steps, independent of `save_steps`. This prevents data loss on crash while allowing large buffer sizes for performance.
- **`buffer_size`**: Each output file gets its own buffer of this size (via `setvbuf` with `_IOFBF`). Default 256 KB. The buffer is `malloc`'d in `init_output_files()` and passed to `setvbuf` as a user-supplied buffer; glibc does not free such buffers on `fclose`, so they live until process exit.

### 8.4 Output Directory (Hash-Based Folders)

When `output_dir = 1`, the program:
1. Reads the ini file content into a string
2. Reads the coupling matrix file (sparse preferred, dense fallback) into a string
3. Computes FNV-1a 64-bit hash of: ini_content + seed_str + matrix_content
4. XORs the pth command-line argument into the hash character by character
5. Creates `results/run_<16-char-hex>/`
6. Copies ini and matrix file(s) into the folder
7. Writes all output to the folder
8. Writes `metadata.txt` with timestamp, parameters, H_final, t_final, stop reason

The FNV-1a constants:
- offset_basis: `14695981039346656037ULL`
- prime: `1099511628211ULL`

---

## 9. Memory Scaling Analysis

### 9.1 Always-Allocated Core Arrays

These are allocated in `allocate_system()` for every run:

| Array(s) | Shape | Size (bytes) |
|----------|-------|:------------:|
| `S_re`, `S_im` | N × D | 2 × N·D × 8 = 16·N·D |
| `DS_re`, `DS_im` | N × D | 2 × N·D × 8 = 16·N·D |
| `JS_re`, `JS_im` | N × D | 2 × N·D × 8 = 16·N·D |
| `S2`, `DS2` | N | 2 × N × 8 = 16·N |
| `h`, `hmod`, `phi` | N | 3 × N × 8 = 24·N |
| `cos_phi`, `sin_phi`, `inv_sqrtS2` | N | 3 × N × 8 = 24·N |
| `Y_re`, `Y_im` | N × 2 | 2 × N·2 × 8 = 32·N |
| `DY_re`, `DY_im` | N × 2 | 2 × N·2 × 8 = 32·N |

Sum: **48·N·D + 128·N** (D-dependent + D-independent terms)

### 9.2 Always-Allocated Sparse Structures

Built once after coupling matrix generation:

| Array | Type | Size (bytes) |
|-------|------|:------------:|
| `csr_row_ptr` | int | 4·(N+1) |
| `csr_col_idx` | int | 4·nnz |
| `csr_Jr` | double | 8·nnz |
| `csr_Ji` | double | 8·nnz |
| `edge_a` | int | 2·nnz (each edge appears once) |
| `edge_b` | int | 2·nnz |
| `edge_Jr` | double | 4·nnz |
| `edge_Ji` | double | 4·nnz |
| **Sparse total** | | **4·N + 4 + 32·nnz** |

For `nnz = k·N` (mean degree k): sparse total ≈ `(4 + 32k)·N`.

### 9.3 Dense J (conditional on compute_eigenvalues)

| Array | Size (bytes) |
|-------|:------------:|
| `J_re` | 8·N² |
| `J_im` | 8·N² |
| **Total** | **16·N²** |

### 9.4 Temporary Peak Allocations

| Function | Size (bytes) | Trigger |
|----------|:------------:|---------|
| `Linearized_Max_Eigenvalue` | 32·N² | `compute_eigenvalues=1` |
| `h_threshold` | 32·N² | `pth >= 0` |
| `initialize_J_Hermitian_Lattice` (edge_decisions) | 24·d·N | `new_coupling=2` |
| I/O buffers | `buffer_size × n_files` | always (max ~1.5 MB) |

All temporaries are freed before the main loop begins (except I/O buffers, which persist).

### 9.5 Three Scaling Cases

The memory formulas below predict the *allocated* heap for user data structures.
Measured heap (`mallinfo2`) is always slightly higher due to glibc malloc overhead
(chunk headers, tcache, arena fragmentation). A **safety margin of 512 KB** is
added to each formula to guarantee that the predicted value is a conservative
overestimate at all tested sizes (see §9.8).

#### Common terms

```
Core arrays (allocate_system):
  D-dependent:  S_re, S_im, DS_re, DS_im, JS_re, JS_im    →  6 × 8·N·D  = 48·N·D
  D-independent (norms, pump, trig):
    S2, DS2, h, hmod, phi, cos_phi, sin_phi, inv_sqrtS2   →  8 × 8·N    = 64·N
    Y_re, Y_im, DY_re, DY_im (2 components each)          →  4 × 16·N   = 64·N
  Core total = 48·N·D + 128·N

Sparse structures (built once, persistent):
  CSR:  csr_row_ptr  → 4·(N+1),  csr_col_idx/Jr/Ji → 20·nnz
  Edge: edge_a/b/Jr/Ji → 12·nnz     (num_edges = nnz/2 undirected edges)
  Sparse total = 4·N + 4 + 32·nnz

Combined: 48·N·D + 128·N + 4·N + 4 + 32·nnz = 48·N·D + 132·N + 4 + 32·nnz
```

Note: `nnz` counts **both directions** of each edge in the CSR matrix (the Hermitian
mirror is stored explicitly). For a graph with E undirected edges, `nnz = 2·E`.

#### Case 1: Random Graph, Sparse Storage (new_coupling=1, compute_eigenvalues=0)

```
E₁ = prob_coup · N · (N-1) / 2    (number of undirected edges)
nnz = 2·E₁

M₁_startup = 48·N·D + 132·N + 4 + 64·E₁ + 524288
M₁_steady  = 48·N·D + 132·N + 4 + 64·E₁ + I/O_buffers + 524288
```

Startup is measured after the rand_r 3-pass completes, CSR/edge list are built,
and temporary arrays (row_count, cur_row = 8·N total) are freed. Steady adds
I/O buffer allocation in `init_output_files` (depends on `save_dyn`; see §8.2).

#### Case 2: Dense File Input (new_coupling=0, compute_eigenvalues=0)

No dense J arrays are allocated — the file is read element-by-element via `fscanf`
and sparse structures are built directly.

```
E₂ = number of undirected non-zero off-diagonal edges
nnz = 2·E₂

M₂_steady = 48·N·D + 132·N + 4 + 64·E₂ + I/O_buffers + 524288
```

For a fully dense matrix (prob_coup=1, E₂ ≈ N²/2):
```
M₂_steady ≈ 48·N·D + 132·N + 4 + 32·N² + I/O_buffers + 524288
```

#### Case 3: Lattice (new_coupling=2, compute_eigenvalues=0)

```
E₃ = N · d   (PBC, prob_coup=1; fewer for OBC or prob_coup<1)
nnz = 2·E₃

M₃_startup = 48·N·D + 132·N + 4 + 64·d·N + 524288
M₃_steady  = 48·N·D + 132·N + 4 + 64·d·N + I/O_buffers + 524288
```

Note: the lattice generator allocates `edge_decisions` (24·d·N bytes) during
construction, but this is freed before the startup measurement point.

### 9.6 Key Scaling Insight

| Connectivity model    | nnz scaling | N=10⁶ memory (D=2)  | Feasible? |
|-----------------------|:-----------:|:-------------------:|:---------:|
| Fully connected (p=1) | N²          | 32 TB (J alone)     | ❌ No     |
| Fixed p=0.1           | 0.1·N²      | 3.2 TB              | ❌ No     |
| p=10/N (k=10)         | 10·N        | 548 MB              | ✅ Yes    | 
| 2D lattice (k=4)      | 4·N         | 356 MB              | ✅ Yes    |
| 3D lattice (k=6)      | 6·N         | 420 MB              | ✅ Yes    |

The critical insight: **sparse storage alone does not fix O(N²) scaling** when prob_coup is constant — it only eliminates the 16·N² dense J overhead (32·nnz vs 16·N² at p=0.7: 22.4·N² vs 16·N², actually worse). The real value of the sparse architecture is:

1. **No 32·N² eigenvalue scratch** (kills dense mode above N ≈ 10⁴)
2. **No O(N³) GSL diagonalization** (hours at N=10⁴, impossible at N=10⁵)
3. **O(N) topologies** (lattices, sparse random graphs with fixed mean degree) become feasible up to 10⁶–10⁸ hyperspins

### 9.7 Practical Limits

| Hardware    | Memory | Max N (all-to-all) | Max N (k=10 sparse) | Max N (2D lattice) |
|-------------|:------:|:------------------:|:-------------------:|:------------------:|
| Laptop      | 16 GB  | 10³                | 5×10⁷               | 7×10⁷              |
| Workstation | 128 GB | 3×10³              | 4×10⁸               | 5×10⁸              |
| HPC node    | 1 TB   | 8×10³              | 3×10⁹               | 4×10⁹              |

### 9.8 Empirical Validation

Four benchmark regimes were tested to validate the memory formulas against
measured heap usage across 8 system sizes (N=10 to 20000) with 10 random
seeds each, for a total of 320 individual runs:

| Regime | E₁ scaling | E₁(N=20000) | Steady predicted (with margin) |
|:------:|:----------:|:------------:|:------------------------------:|
| p=0.1  | O(N²)      | ~20 M        | 1.28 GB                        |
| p=0.5  | O(N²)      | ~100 M       | 6.40 GB                        |
| p=4/N  | O(N)       | ~40 K        | 7.9 MB                         |
| p=8/N  | O(N)       | ~80 K        | 10.5 MB                        |

Heap measurements use `mallinfo2()` (uordblks + hblkhd) captured at two
points in `initialize_J_Hermitian_Binary_Sparse`:
- **startup_heap_B**: set after all temporary coupling-generation arrays
  are freed (after the rand_r 3-pass completes and CSR/edge list are built)
- **steady_heap_B**: set after I/O buffer allocation in `init_output_files`

Resident set size (RSS) is obtained via `getrusage(RUSAGE_SELF)`.

**Protocol:** For each regime, prob_coup is fixed (for p=0.1, p=0.5) or set
to 4/N or 8/N (for sparse regimes) with `new_coupling=1`, `type_coup=2`, `D=2`,
`compute_eigenvalues=0`. Ten independent matrix realizations per size.
Each run is a complete simulation: coupling generation → I/O init → 10 RK2
steps → output (the dynamics is not run to convergence; only memory
measurement matters).

**Results files** (in docs/Memory_Usage_Tests/):
- `mem_p01.dat` — prob_coup=0.1
- `mem_p05.dat` — prob_coup=0.5
- `mem_pN4.dat` — prob_coup=4/N
- `mem_pN8.dat` — prob_coup=8/N

Each file format: `N  measured_startup  predicted_startup  measured_steady
predicted_steady  resident_set_size` (space-separated, 1 header line + 80
data lines, 10 per N). The `predicted_startup` and `predicted_steady`
columns use the corrected formulas of §9.5 (132·N + 512 KB safety margin;
132·N was deliberately chosen over the earlier 152·N/160·N variants, and
the margin ensures the formula overestimates — never underestimates —
across all sizes).

**Steady-state validation:**

```
M₁_steady = 48·N·D + 132·N + 4 + 64·E₁ + I/O_buffers + 524288
```

where E₁ = prob_coup · N · (N-1) / 2 (expected value, not actual edge
count) and 524288 bytes (512 KB) is the safety margin for glibc malloc
overhead. At N=20000 (worst-case errors across 10 seeds):

| Regime | Measured (avg) | Predicted (132N+margin) | Error |
|:------:|:--------------:|:-----------------------:|:-----:|
| p=0.1  | 1.285 GB       | 1.285 GB                | −0.04% |
| p=0.5  | 6.405 GB       | 6.405 GB                | −0.01% |
| p=4/N  | 7.48 MB        | 7.91 MB                 | −5.4%  |
| p=8/N  | 10.04 MB       | 10.47 MB                | −4.1%  |

All errors are negative → the formula **always overestimates** (safe).
The relative error shrinks with N because the 512 KB margin is a fixed
overhead while the formula value scales with N (or N²). At N=1000 the
overestimate is larger (up to ~82% for O(N) regimes) because the 512 KB
margin dominates the small total; this is acceptable since the absolute
memory is tiny (≤ 1 MB). At N=20000 the worst case is ~5%, and at N=10⁶
it would be under 0.1%.

**Startup validation:**

The startup formula `M₁_startup = 48·N·D + 132·N + 4 + 64·E₁ + 524288`
is validated only for the random graph generator (new_coupling=1).
The measured value is captured after the coupling matrix is fully built
and all temporary arrays (row_count, cur_row) have been freed.
At N=20000: measured startup ranges from 1.2845 GB to 1.2849 GB (p=0.1),
while the formula predicts 1.2850 GB — always overestimating by ≤ 0.03%.

**Why the safety margin is needed:**

The analytical formula describes the *allocated* user data structures only.
Measured heap (`mallinfo2().uordblks + hblkhd`) is always higher due to:
- Per-allocation metadata (chunk headers, ~16-32 bytes each × 34 arrays)
- tcache per-thread cache (glibc 2.26+, 64 KB per thread × up to 64 bins)
- Arena fragmentation with multiple malloc arenas

The overhead is:
- **At small N (≤ 100):** the formula overestimates measured heap by 2-3× (the
  512 KB margin dwarfs the ~300 KB no-margin prediction). Measured heap itself
  tracks the no-margin formula to within ~1.04×.
- **At N = 1000:** overhead is ~10 KB. The 512 KB margin is a generous buffer.
- **At N ≥ 10000:** overhead plateaus at ~100 KB. The 512 KB margin contributes
  ≤ 0.04% for O(N²) and ~5% for O(N).
- **At N ≥ 10⁶:** overhead remains ~100-150 KB (dominated by per-mmap page
  alignment for the 26 core allocations, not by N). The margin contributes
  < 0.1% even for the sparsest O(N) regimes.

Because the 512 KB margin is a *fixed* overhead while the formula value
scales as N (O(N) regimes) or N² (O(N²) regimes), the relative error
shrinks as 1/N or 1/N² and becomes negligible at large N. The 512 KB
value was chosen as the minimum that guarantees overestimation across all
320 benchmark runs (8 sizes × 10 seeds × 4 regimes).

**RSS vs heap:**

RSS/steady heap is 7-11× at N ≤ 100 (the fixed ~3 MB process overhead of stack, code, and libraries dominates the tiny heap) and drops to ~1.0-1.1× at N = 20000 as the heap dominates. The ratio converges toward ~1.0× at large N.

**Known limitation:**

startup_heap_B is only recorded for the random graph generator
(`initialize_J_Hermitian_Binary_Sparse`). Lattice mode
(`new_coupling=2`) does not set this field. A future update could add the
same `mallinfo2` snapshot to the lattice generator.

---

## 10. Parallelization

### 10.1 What Is Parallelized

All functions with independent per-field or per-edge work use OpenMP, gated by a global `use_parallel` flag:

| Parallel region | Loop over | Description |
|----------------|-----------|-------------|
| `compute_coupling_product_flat` | fields a | Each thread processes a disjoint set of rows |
| `RK2_step` (midpoint phase) | fields a | Per-field derivative computation |
| `RK2_step` (update phase) | fields a | Per-field full-step update |
| `compute_observables` (inv_sqrtS2) | fields i | Precompute `1/√|S|²` for normalization — only when `compute_H=1` |
| `compute_observables` (field pass) | fields a | L0, DS² reduction — only when `compute_full=1` |
| `compute_observables` (edge pass) | edges | H + Lc reduction — only when `compute_H=1`; Lc further gated by `compute_full` |
| `compute_observables` (magnetization) | fields a | VLA array reduction `compR[:D],compI[:D]` — only when `compute_full=1` |

### 10.2 Thread Safety

All parallel regions are thread-safe because:
1. **Disjoint array slices**: Each thread writes to a distinct range of array indices (e.g., `a` from `tid*chunk` to `(tid+1)*chunk`)
2. **OpenMP reduction**: Shared accumulators (L₀, Lc, H, m, mean_S2, mean2_S2) use OpenMP's built-in `reduction(+:var)` clause
3. **Read-only shared data**: The coupling matrix (CSR + edge list) and pump values are never modified during parallel execution
4. **Local variables**: Per-iteration variables (coefficients, accumulators) are stack-local per thread

### 10.3 When Parallelization Pays Off

Each save step / E-target step / stability-check step / final step creates 8 parallel regions (coupling product 2×, RK2 midpoint,
RK2 update, inv_sqrtS2, field pass for L0/DS², edge pass for H/Lc, magnetization). Idle steps (no save, no E-target, no check, no final; typical `save_dyn=0` benchmarks) create only 4
(coupling product 2×, RK2 midpoint, RK2 update — the `compute_H` gate skips the observables entirely). Each region pays a ~5 µs overhead
for thread fork, work distribution, and implicit barrier. Total: ~20 µs/step
of pure coordination cost on idle steps (4 always-active × 5 µs).

Parallelization only helps when the actual computation dwarfs this overhead.

**Break-even formula:**

```
N_crit(k, D, n_t) = 58000 · n_t/(n_t-1) / [(21k + 32)·D + 100]
```

**Understanding each term — Numerator:**

  `58000`  =  F_eff × n_regions × τ
              │         │           └── τ = 5 µs: barrier + fork cost per region
              │         │               (warm thread pool, Linux, schedule(static))
               │         └── 6: always-active parallel regions per observables step
               │               (8 total on save steps, 2 gated by compute_full:
               │                field pass for L0/DS² and magnetization)
               │
               │   v2.9 note: this models the worst-case step — one that computes H
               │   (save, E-target, stability-check, or final). On the idle steps
               │   between them the compute_H gate skips observables entirely, so only
               │   4 regions run (coupling product 2×, RK2 midpoint, RK2 update):
               │   ~20 µs/step coordination, and the 5·k·D edge-pass and inv_sqrtS2
               │   flops below are also absent. N_crit is therefore a conservative
               │   upper bound for save_dyn=0 / long-gap runs; their true crossover
               │   is roughly 2/3 of the quoted value.
              └── F_eff ≈ 2×10^9 flops/s: effective single-core throughput.
                  This is NOT peak flops — for sparse CSR operations the CPU
                  spends most of its time waiting on memory, not computing.
                  2 GFLOP/s is realistic for this access pattern.

The raw product 2×10⁹ × 6 × 5×10⁻⁶ = 60000; the constant is calibrated down
to 58000 to match the empirically observed crossover (the components are
estimates, and the constant absorbs their combined uncertainty).

  `n_t/(n_t-1)` accounts for the Amdahl serial fraction within the parallel
  work. When the parallel region ends, one thread's last chunk serializes:
    n_t=2  → 2.00   (half the "parallel" time is effectively serial)
    n_t=4  → 1.33
    n_t=8  → 1.14   (only 1/8 of parallel time is serial — good)
    n_t=16 → 1.07

**Understanding each term — Denominator (work per field, flops):**

  `21k·D`  Coupling-related work. Each field interacts with k neighbors:
             - Coupling product (J×S via CSR): 8 flops × 2 phases × k × D
               = 16·k·D per field. This is the dominant cost for large k.
             - Edge pass for Hamiltonian: ~10 flops per edge × (k/2) edges
               per field × D = 5·k·D per field.
           Sum: 21·k·D. For dense matrices (k → N) this term causes O(N²)
           scaling and makes parallelization trivially worthwhile.

  `32·D`   RK2 component loop work. Both phases (midpoint + update) process
           D components per field with ~16 flops per component per phase:
             - f0R/f0I (non-interacting):   ~6 flops/comp
             - fLR/fLI (coupling, read from precomputed JS): ~2 flops/comp
             - feqR/feqI (equalization):     ~2 flops/comp
             - DS_RE/DS_IM update step:      ~6 flops/comp
           × 2 phases = 32·D. This always scales with D regardless of k.

  `100`    Fixed per-field cost, independent of k and D:
             - RK2 setup per phase (coefficients, trig):       ~29 flops
             - RK2 equalizer derivatives per phase:            ~16 flops
             - inv_sqrtS2 (1/sqrt for normalization to hyperspin): ~10 flops
           Two phases: 2×(29+16) + 10 ≈ 100.

**Calibration:**

The constant 58000 is calibrated to the empirically observed crossover:
at k=4 (2D lattice), D=2, n_t=8, the parallelization becomes marginal
around N≈200. The formula gives:

  N_crit = 58000 · 1.14 / [(84+32)·2 + 100] = 66120 / 332 ≈ 200  ✓

**Reference table (D=2):**

  k        Example                     n_t=4    n_t=8    n_t=16
  ─────    ───────                     ─────    ─────    ──────
  4        2D lattice (k=4)             233      200      186
  6        3D lattice (k=6)             186      159      149
  10       Sparse random (k=10)         132      114      106
  20       Moderately connected          77       66       62
  50       Dense-ish                     34       29       27
  100      Nearly all-to-all             18       15       14

**Dependence on D:**

Higher D multiplies the per-field work linearly. For k=4:

  D=2  → N_crit ≈ 200
  D=3  → N_crit ≈ 200 × (21×4+32)×2+100 / (21×4+32)×3+100
          = 200 × 332 / 448 ≈ 148
  D=4  → N_crit ≈ 200 × 332 / 564 ≈ 118

**Practical guideline:**

  N < N_crit/2    →  num_threads = 1       (overhead dominates)
  N_crit to 2·N_crit  →  benchmark both   (marginal, try 4 threads)
  N > 2·N_crit   →  use all cores          (clear speedup expected)

The old "N > 50" rule was only valid for fairly dense matrices (k ≳ 75-80,
where N_crit ≈ 20; at k=150 N_crit has already dropped to ≈ 10). For our 2D
lattices (k=4) the real threshold is ~200.

The `if(use_parallel)` clause in every `#pragma omp parallel for` ensures
zero OpenMP overhead when `num_threads <= 1` — the code compiles and runs
as pure sequential even with `-fopenmp`.

### 10.4 Determinism

Both coupling generation paths (`new_coupling=1` and `new_coupling=2`) are fully deterministic for a given NS. The random graph path uses `rand_r` with a local seed, while the lattice path uses `rand()` seeded by `srand(NS)`. For cross-version comparison (e.g., v2.1 vs v2.2+), always load matrices from file (`new_coupling=0`). With file-loaded matrices and tight convergence criteria (`allowed_H_change=1e-15`), the dynamics converge to the same final energy regardless of thread count, machine, or compiler.

---

## 11. Scripts & Workflows

For batch benchmarks, convergence tests, and visualization, see the dedicated scripts documentation:

```
docs/SCRIPTS_README.md
```

This covers:
- **complete_bench.sh**, **Convergence_TtS.sh**, **Hmin_bench.sh**, **merge_bench_results.sh**: Legacy scripts moved to `scripts/legacy/`. Superseded by `bench_family.sh` and `bench_family_append.sh`.
- **plot_hmin_histogram.sh**: Stacked histogram of Hmin relative errors.
- **Plotting tools**: `plot_lattice.x` (2D lattice visualization), `plot_circular.x` (connectogram).
- **Format converters**: `matrix_dense_to_sparse.x`, `matrix_sparse_to_dense.x`.
- **Diagnostics**: `analyze_state.x` — bond satisfaction analysis for post-run validation.

Dependencies for scripts: `Complex_HSM.x`, `Complex_HSM.ini`, GSL, OpenMP, standard shell utilities (awk, sed, grep, date), gnuplot 6.0+ with pngcairo (for histogram and plotting tools).

---

## 12. Complete Function Reference

### 12.1 Configuration & Input

#### `parse_config`
```c
int parse_config(const char *filename, int *N, int *D, ..., int *lattice_bc);
```
Reads all parameters from `Complex_HSM.ini`. Sets defaults for optional parameters: `buffer_size=262144`, `flush_interval=1000`, `num_threads=1`, `stable_window=1000`, `relative_H_change=0`, `steps_for_update=1`, `E_target_tol=1e-10`, `save_steps=1`, `compute_eigenvalues=1`, `lattice_d=2`, `lattice_NL=10`, `lattice_bc=1`, `initial_type=2`. Skips `#` comments and empty lines. Splits on `=`, trims whitespace, matches keys via `strcmp`. Returns 0 on success, 1 on file error.

#### `print_header_and_ini_parameters`
```c
void print_header_and_ini_parameters(const char *filename, int N, int D, ..., int lattice_NL, int lattice_bc);
```
Prints the ASCII banner and all configuration parameters to stdout. Organized by parameter group (system, cavity, coupling, time, pump, end conditions, output, parallelization). Includes conditional messages for different `new_coupling` modes, `compute_eigenvalues`, and `save_dyn` settings. Takes `initial_type` (reported in the cavity group alongside `Amax`).

### 12.2 Memory Management

#### `allocate_system`
```c
int allocate_system(int N, int D, system_state *s);
```
Allocates 18 flat arrays via `calloc` (zeroed memory): `S_re`, `S_im`, `S2`, `DS_re`, `DS_im`, `DS2`, `Y_re`, `Y_im`, `DY_re`, `DY_im`, `h`, `hmod`, `phi`, `JS_re`, `JS_im`, `cos_phi`, `sin_phi`, `inv_sqrtS2`. Dense `J_re`/`J_im` are set to NULL and `J_allocated = 0`. Returns 1 if any allocation fails, 0 on success.

#### `free_system`
```c
void free_system(system_state *s);
```
Frees all arrays (including CSR, edge list, and dense J if `J_allocated`). Sets the struct to zero via `memset` to prevent dangling pointers.

### 12.3 Initialization

#### `initialize_fields`
```c
void initialize_fields(system_state *s, double Amax, int initial_type);
```
Random initialization: each field component (Re/Im) and each equalizer (Re/Im) is set to `uniform[-Amax, +Amax]` using `rand()` seeded via `srand(NS)` in main(). N×D fields + N×2 equalizers × 2 components each. `initial_type` constrains BOTH the S fields and the Y0/Y1 equalizers: 0 forces Im ≡ 0, 1 forces Re ≡ 0, 2 leaves both general. Both Re and Im are always drawn from `rand()` first (stream preservation, see §5.6), then the unused component is zeroed.

#### `initialize_J_Hermitian_Binary_Sparse`
```c
void initialize_J_Hermitian_Binary_Sparse(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup, int NS);
```
Generates a random sparse Hermitian coupling matrix directly into CSR + edge list using a `rand_r` 3-pass algorithm (see §5.3). No O(N²) temporary storage. Coupling strength = `1/(N·prob_coup)` scaled by `type_coup`. Writes `Data_Matrix_sparse.dat` via `write_sparse_matrix_file()`. Records `startup_heap_B` from `mallinfo2` after allocation.

#### `initialize_J_Hermitian_Lattice`
```c
void initialize_J_Hermitian_Lattice(system_state *s, int lattice_d, int lattice_NL, int lattice_bc, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup);
```
Generates a d-dimensional hypercubic lattice coupling matrix directly into CSR + edge list (see §6.2). Uses a temporary `edge_decisions` array (24·d·N bytes). Validates N = NL^d. Writes `Data_Matrix_sparse.dat`. If `s->J_allocated`: also writes `Data_Matrix.dat` via `write_dense_matrix_file()`.

#### `initialize_J_Hermitian_Binary_Random`
```c
void initialize_J_Hermitian_Binary_Random(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup);
```
Legacy dense-mode random generator (v2.0 era): builds J directly in dense `J_re`/`J_im` and writes only `Data_Matrix_dat`. Superseded by the sparse-first `initialize_J_Hermitian_Binary_Sparse`; retained but unused by the current initialization paths.

#### `apply_coupling_rotation`
```c
void apply_coupling_rotation(system_state *s, double coup_phase);
```
Applies a global rotation `e^{i·coup_phase}` to every coupling element. Updates all three storage formats: edge list, CSR, and dense J (if allocated). Called once in `main()` after matrix construction for all paths (random graph, lattice, file-loaded). When `coup_phase = 0`, returns immediately (no-op). Complexity: O(E × avg_degree).

#### `initialize_J_Read_From_File`
```c
void initialize_J_Read_From_File(system_state *s, const char *filename);
```
Legacy dense-mode file reader: reads a dense N×N matrix file into `J_re`/`J_im`. Superseded by `initialize_J_Sparse_From_File` (dense input) and `initialize_J_Sparse_From_Sparse_File` (edge-list input); retained but unused by the current initialization paths.

#### `initialize_J_Sparse_From_File`
```c
void initialize_J_Sparse_From_File(system_state *s, const char *filename);
```
Reads dense N×N matrix file into CSR + edge list. Two-pass file read: first pass counts non-zero off-diagonal entries per row, second pass fills CSR and edge list. O(N²) file read time, O(nnz) storage.

#### `initialize_J_Sparse_From_Sparse_File`
```c
void initialize_J_Sparse_From_Sparse_File(system_state *s, const char *filename);
```
Reads sparse edge-list file (`Data_Matrix_sparse.dat`) into CSR + edge list. Two-pass: first pass counts per-row degree from edges, second pass fills. O(nnz) time and memory. The file stores (a < b) with Jr + i·Ji = J(a,b); the Hermitian conjugate J(b,a) = Jr − i·Ji is derived during CSR fill.

#### `fill_dense_from_sparse`
```c
void fill_dense_from_sparse(system_state *s);
```
Copies CSR/edge list data into pre-allocated dense `J_re`/`J_im` arrays. Zeros the arrays first, then iterates the edge list. O(nnz) time. Only called when `compute_eigenvalues=1`.

#### `write_sparse_matrix_file`
```c
void write_sparse_matrix_file(system_state *s, const char *filename);
```
Writes edge-list format from `edge_a/b/Jr/Ji`. First line: `num_edges`. Subsequent lines: `a  b  Jr  Ji`. O(nnz) time, O(1) memory.

#### `write_dense_matrix_file`
```c
void write_dense_matrix_file(system_state *s, const char *filename);
```
Writes dense N×N grid from `J_re`/`J_im`. O(N²) time. Only available when `s->J_allocated = 1`.

#### `build_sparse_coupling`
```c
void build_sparse_coupling(system_state *s);
```
Legacy function: reads dense `J_re`/`J_im` arrays and builds CSR + edge list. Two-pass dense scan. Still available but no longer used by the initialization paths (sparse-first builds directly).

#### `initialize_pump`
```c
void initialize_pump(system_state *s, double *target, double hm, double dh);
```
Sets per-field values: `target[i] = hm + uniform[-dh, dh]`. Used for both h (pump amplitude) and phi (pump phase), called separately for each.

#### `modify_pump`
```c
void modify_pump(system_state *s);
```
Adds pre-computed offset `hmod[i]` to all `h[i]`. Called when t = t_pump. The cumulative offset is `h[i] = h[i] + hmod[i]` where `hmod[i] = Dh + 0·uniform` (set in main via `initialize_pump(&s, s.hmod, Dh, 0)`).

#### `precompute_trig`
```c
void precompute_trig(system_state *s);
```
Computes `cos(phi[a])` and `sin(phi[a])` for all fields into the scratch arrays `cos_phi`/`sin_phi`. Called once at startup (right after `initialize_pump`), since `phi` never changes during the run. Avoids 4·N trig calls per step — the RK2 core never calls `cos()`/`sin()` itself.

#### `compute_hyperspin_norm`
```c
void compute_hyperspin_norm(system_state *s);
```
Computes the squared field magnitude `|S_a|² = Σ_j (Re² + Im²)` for all hyperspins into `S2[]`. These norms are used to normalize fields into hyperspins (`Ŝ = S/|S|`) via the `inv_sqrtS2` buffer in observable computations. Called after initialization and at the end of each RK2 step.

### 12.4 Dynamics

#### `compute_coupling_product_flat`
```c
void compute_coupling_product_flat(const system_state *s,
    const double *src_re, const double *src_im,
    double *out_re, double *out_im);
```
Computes the J×S matrix-vector product for all fields: `out[a*D+j] = Σ_b J_{ab} · src[b*D+j]`, where the complex multiplication is `(Jr + i·Ji)·(sr + i·si) = (Jr·sr − Ji·si) + i·(Jr·si + Ji·sr)`. Uses CSR format (directed edges). Parallel OpenMP over the outer field loop a. Called once per RK2 phase (once with `S`, once with `DS`).

#### `RK2_step`
```c
void RK2_step(system_state *s, double dt, double g, double beta, double deq, int coup_control, int eq_control);
```
Advances all fields and equalizers by dt using RK2 midpoint rule. Two phases (see §3.1 for full algorithm). Within each phase, per-field computations are parallelized via OpenMP. Each field a accesses its left neighbor `(a+N-1)%N` and right neighbor `(a+1)%N` on the ring to compute equalizer forces from the imbalance difference `Y²(a) − Y²(left)`.

**Midpoint equalizer derivative:**
```c
dY0R = 0.5 * deq * (-S2[a] + S2[next]) * Y_re[a][0];
dY1R = -0.5 * deq * (-S2[a] + S2[next]) * Y_re[a][1];  // opposite sign
```

**Update phase equalizer derivative:**
Uses midpoint values (DY and DS2) instead of current (Y and S2), same structure.

### 12.5 Observables

#### `S2_of_spin`
```c
double S2_of_spin(system_state *s, int a);
```
Returns `|S_a|² = Σ_j (Re² + Im²)` for a single field.

#### `Y2eq_of_spin`
```c
double Y2eq_of_spin(system_state *s, int a);
```
Returns `|Y_a₀|² − |Y_a₁|²` (equalizer imbalance).

#### `max_eigenvalue_symm`
```c
double max_eigenvalue_symm(double *data, int n);
```
Wraps GSL `gsl_eigen_symm` to find the largest eigenvalue of an n×n symmetric matrix stored in flat 1D array `data[i*n+j]`. Finds max after extracting all eigenvalues.

#### `Linearized_Max_Eigenvalue`
```c
double Linearized_Max_Eigenvalue(int N, double *J_re, double *J_im, double *h, double g);
```
Builds the 2N×2N linearized dynamics matrix (see §2.5) on the heap via `calloc`, finds its largest eigenvalue via `max_eigenvalue_symm`, frees the matrix, returns the eigenvalue. The `h` parameter allows computing both `Jmax` (with h=0, g=0) and `Lmax` (with actual h and g).

Usage in main():
- `Linearized_Max_Eigenvalue(N, J_re, J_im, hmod, 0)` → Jmax (no pump, no loss)
- `Linearized_Max_Eigenvalue(N, J_re, J_im, h, g)` → Lmax (with pump and loss)

#### `h_threshold`
```c
double h_threshold(int N, double g, double *J_re, double *J_im);
```
Scans h from 0 upward in steps of 0.00001, computing the max eigenvalue of the linearized matrix at each step (reusing a single `calloc`'d buffer). Returns the first h where the eigenvalue exceeds 0. Prints `hth` and `lmax` to stdout.

#### `Ly0`, `Lyc`, `Hamiltonian`, `magnetization`, `Average_Amplitude_Variance`

These functions are no longer standalone — all observables are computed inside `compute_observables()` (see §12.6). The formulas are unchanged:

`Ly0` (non-interacting Lyapunov):
```
L0 = −½·Σ_a [ ½·h_a·Σ_k(Re²−Im²) − g·|S_a|² − ½β·|S_a|⁴ ]
```

`Lyc` (coupling Lyapunov, per edge — the ½ in the formula cancels the 2× from ordered-pair double-counting):
```
Lc -= Jr*(S_RE(a,k)*S_RE(b,k) + S_IM(a,k)*S_IM(b,k))
    +  Ji*(S_IM(a,k)*S_RE(b,k) - S_RE(a,k)*S_IM(b,k));
```

`Hamiltonian` (hyperspin energy, edge list, ×2):
```
H = −2·Σ_{p,q,k} [ Jr·(Ŝ_p·Ŝ_q) − Ji·(Ŝ_p × Ŝ_q) ]
```
where `Ŝ_a = S_a / |S_a|`.

`magnetization`:
```
<m> = (1/N)·sqrt( Σ_k |Σ_a Ŝ_{ak}|² )
```

`Average_Amplitude_Variance`:
```
Var = <S₂²> − <S₂>², returned as |Var|/N
```

#### `DLyR`

Removed — the time derivative `dL/dt = (L(t) − L(t−dt)) / dt` is computed inline inside `compute_observables()`.

### 12.6 Output Management

#### `init_output_files`
```c
io_context init_output_files(int N, int D, int save_dyn, int buffer_size, const char *base_dir);
```
Opens files based on `save_dyn` mode. The `base_dir` parameter prefixes all output paths (empty string = current directory). Writes headers with column labels. Sets up buffered I/O with `setvbuf(malloc(buffer_size))`. Returns `io_context` struct with `FILE*` handles (NULL for unopened files).

#### `compute_observables`
```c
observables compute_observables(system_state *s, double g, double beta, double dt, double *L_old, int compute_H, int compute_full);
```
Computes observables from the current system state with merged loops to reuse intermediate quantities. The `compute_H` and `compute_full` flags implement a two-level gate (see §8.3):

- **`compute_H=0, compute_full=0`** (idle steps): the entire call is skipped in the main loop — no observables computed.
- **`compute_H=1, compute_full=0`** (E-target / stability-check / final steps): `inv_sqrtS2` is precomputed, then the edge pass accumulates H + Lc (L0, magnetization, DS² are **not** computed).
- **`compute_full=1`** (save steps): three merged passes — **(1)** single field pass computing L0, mean_S2, and mean2_S2 jointly; **(2)** edge pass computing H and Lc jointly; **(3)** magnetization pass (parallelized over N fields via OpenMP 4.5 VLA array reduction) reusing `inv_sqrtS2` from pass 1. L = L0 + Lc and dL = (L - L_old)/dt are computed after the passes.

Returns an `observables` struct; uncomputed fields are set to 0. See §8.3.

#### `save_output_files`
```c
void save_output_files(io_context *ctx, int save_dyn, system_state *s, double t, double L, double L0, double Lc, double dL, double H, double m, double DS2);
```
Writes current observables to open files. Called after `compute_observables()` when `steps % save_steps == 0` and `save_dyn > 0`.

- `save_dyn=1`: writes `Data_Fields.dat` (N rows), `Data_Equalizers.dat` (1 row), `Data_Hamilton.dat` (1 row), `Data_Lyapunov.dat` (1 row)
- `save_dyn=2`: writes `Data_Hamilton.dat` and `Data_Lyapunov.dat`
- `save_dyn=3`: writes `Data_H_Ly_m_DS2.dat` (1 row with t, H, Ly/N, m, DS²)

#### `flush_output_files`
```c
void flush_output_files(io_context *ctx, int save_dyn);
```
Forces buffered data to disk via `fflush` for all open files. Called every `flush_interval` steps.

#### `close_output_files`
```c
void close_output_files(io_context *ctx, int save_dyn);
```
Closes all open file handles with NULL safety checks.

#### `save_final_config`
```c
void save_final_config(io_context *ctx, system_state *s, double t, int NS);
```
Writes final field state to `Data_Final.dat`. Format: `NS  T  S1  Re(A11)  Im(A11)  ...`. All values with 15 decimal places.

### 12.7 Hash & Output Directory Helpers

#### `read_file_to_string`
```c
char *read_file_to_string(const char *path);
```
Reads entire file into a `malloc`'d null-terminated string. Used to feed ini and matrix contents into the hash. Caller must `free()` the returned buffer.

#### `compute_hash`
```c
unsigned long long compute_hash(const char *ini_path, int seed, const char *matrix_path);
```
FNV-1a 64-bit hash: reads ini file into string → hash each byte → hash seed string → hash matrix file bytes. Returns the hash value. Used for deterministic run folder naming.

FNV-1a constants:
- `offset_basis = 14695981039346656037ULL`
- `prime = 1099511628211ULL`

The main function additionally XORs each character of the `pth` command-line argument into the hash after `compute_hash()` returns, ensuring runs with different pth get different folders.

#### `format_hash_hex`
```c
void format_hash_hex(unsigned long long hash, char hex[17]);
```
Converts a 64-bit hash to a 16-character lowercase hex string (LSB first: nibble 0 → hex[15], nibble 15 → hex[0]). Null-terminates the output buffer.

#### `copy_file_to_dir`
```c
int copy_file_to_dir(const char *src, const char *dir, const char *dst_name);
```
Copies a file from `src` to `<dir>/<dst_name>` using an 8 KB buffer. Returns 0 on success, 1 on error.

#### `write_metadata`
```c
void write_metadata(const char *dir, int NS, int N, int D, int initial_type, double pth, double hm, double H, double t_final, const char *stop_reason);
```
Writes `metadata.txt` to the result folder: program version ("Complex_HSM v2.9"), timestamp (YYYY-MM-DD HH:MM:SS), NS/N/D/initial_type/pth/hm, H_final, t_final, stop reason, and a list of output files. Note: the output-file list is hardcoded (`Data_Final.dat`, `Data_Matrix.dat`, `Complex_HSM.ini`) and does not reflect what was actually produced — in sparse-only runs no `Data_Matrix.dat` exists and `Data_Matrix_sparse.dat` is never listed.

### 12.8 Utility Functions

#### `print_memory_peak`
```c
static inline void print_memory_peak(system_state *s);
```
Uses `getrusage(RUSAGE_SELF)` to get max RSS and `mallinfo2()` to get heap usage. Records `steady_heap_B` and prints startup heap, steady heap, and RSS to stdout.

---

## 13. Troubleshooting & FAQ

### 13.1 Compilation Issues

| Error | Likely cause | Solution |
|-------|-------------|----------|
| `gcc: command not found` | Compiler not installed | `sudo apt install gcc` (Debian) or `sudo dnf install gcc` (Fedora) |
| `fatal error: gsl/gsl_eigen.h: No such file or directory` | GSL not installed | `sudo apt install libgsl-dev` or `sudo dnf install gsl-devel` |
| `undefined reference to 'omp_get_thread_num'` | OpenMP flag missing | Recompile with `-fopenmp` |
| `undefined reference to 'gsl_...'` | GSL libraries not linked | Add `-lgsl -lgslcblas` to compile command |

### 13.2 Runtime Issues

| Symptom | Likely cause | Solution |
|---------|-------------|----------|
| `Failed to open config file` | `Complex_HSM.ini` not in cwd | Run from the directory containing the ini, or provide the full path |
| `Error reading Data_Matrix.dat` | `new_coupling=0` but no matrix file | Set `new_coupling=1` to generate one, or place the matrix file in the working directory |
| `Unknown key: ...` | Typo in ini key | The offending key is printed to stderr; check spelling |
| No output files created | Wrong `save_dyn` mode | Check §8 for what each mode produces |
| Convergence never triggers | `allowed_H_change` too strict or `stable_window` too large | Relax tolerance, reduce window size, or set `relative_H_change=1` for N-scalable check |
| Run ends at t = tmax without converging | Not enough time | Increase `tmax` or relax stop conditions |
| `Error: pth >= 0 requires eigenvalue computation` | `pth >= 0` with `compute_eigenvalues=0` | Set `pth = -1` or enable `compute_eigenvalues=1` |
| Program crashes at large N | Memory exhausted | Use `compute_eigenvalues=0` for N > 2000; check system RAM limits |

### 13.3 Physics / Numerical

**Q: How do I get reproducible results?**

Use the same seed and same config. With file-loaded matrices and `allowed_H_change=1e-15`, the dynamics converge to the same final energy regardless of thread count, machine, or compiler. The coupling matrix generation is fully deterministic regardless of thread count.

**Q: What's a good dt?**

RK2 stability requires `dt ≲ C / max(h, g, β·|S|², |λ_J|, deq)` where C ≈ 1–2 (see §7.5). Start with `dt = 0.01` and decrease if the simulation behaves erratically. The step size is fixed — there is no adaptive stepping.

**Q: How do I know if my simulation has converged?**

The program prints `> t_final = ...` to stdout when it stops, along with the stop reason:
- "Target energy reached" — final H ≤ E_target × (1 − E_target_tol)
- "Energy stable" — energy varied less than or equal to `allowed_H_change` for `stable_window` consecutive checks
- "Reached tmax" — max simulation time reached (may not be converged)

A converged run typically stops via the stability condition with small `allowed_H_change`. For large N, use `relative_H_change = 1` so the tolerance scales with system size — otherwise the absolute `allowed_H_change` may be too strict to trigger in practice.

**Q: What does the equalizer do?**

Equalizers drive neighboring fields toward equal magnitudes. Without them, fields can develop extremely disparate amplitudes, leading to numerical instability. `deq` controls how strongly they act. The ring topology means field a equalizes with its immediate neighbors in a chain.

**Q: Why does the program sometimes print `H_final` as 0.000000 or large values?**

Check that the simulation ran long enough for coupling to activate (t > t_coup). If tmax is small and t_coup is large, the system may never feel the coupling. Also verify that `coup_control` and `eq_control` are being activated (they print to stdout implicitly via log timing).

**Q: When should I use h > 0?**

Only use h > 0 for **real** coupling matrices (`type_coup = 0`) where the pump and coupling commute. For complex J (`type_coup = 1` or `2`), always use h = 0 (set `hm = 0` and `pth = -1`).

**Q: What do Jmax and Lmax mean?**

- **Jmax**: largest eigenvalue of the coupling matrix J. Measures the overall coupling strength. Used for diagnostics and time-step estimation.
- **Lmax**: largest eigenvalue of the linearized dynamics matrix (J + pump + loss). When Lmax > 0, the origin is unstable and the system will amplify perturbations. When Lmax < 0, the origin is stable and the system decays.

---

## Appendices

### A. Notation Conventions

| Symbol | Meaning | Example |
|--------|---------|---------|
| N | Number of hyperspins | `N = 100` |
| D | Hyperspin dimension (complex components per hyperspin) | `D = 2` |
| d | Lattice spatial dimension | `lattice_d = 2` (2D square lattice) |
| S_a | Complex hyperspin vector for hyperspin a, size D | `S_RE(s, a, j)` |
| Y_a | Complex equalizers for hyperspin a (2 fields) | `Y_RE(s, a, k)`, k ∈ {0,1} |
| J | N×N Hermitian coupling matrix | `J_RE(s, a, b)` + i·`J_IM(s, a, b)` |
| h_a | Pump amplitude for hyperspin a (real) | `s->h[a]` |
| φ_a | Pump phase for hyperspin a (real) | `s->phi[a]` |
| g | Intrinsic loss rate | `gm` in config |
| β | Nonlinear saturation | `beta` in config |
| deq | Equalization strength | `deq` in config |
| dt | Integration time step | `dt` in config |
| hth | Pump threshold | auto-computed |

**Important:** d and D are independent. A d-dimensional lattice of N = NL^d hyperspins can have any hyperspin dimension D. Do not confuse the two.

### B. Indexing Conventions

- **Hyperspins:** 0-indexed (a = 0, ..., N-1). Output file headers and data rows display field IDs as 1-indexed.
- **Hyperspin components:** 0-indexed (j = 0, ..., D-1).
- **Equalizer components:** k = 0, 1. Y₀ gets the +difference, Y₁ gets the −difference in the ODE.
- **Lattice coordinates:** row-major (C ordering). coord[0] is the fastest-varying dimension.
- **Coupling matrix:** row-major: J(i,j) = row i, column j. Hermitian: J_re symmetric, J_im antisymmetric.
- **Edge list:** stores upper triangle (a < b) for undirected edges. Hermitian conjugate is J(b,a) = Jr − i·Ji.

### C. Coupling Matrix File Formats

#### Sparse format (`Data_Matrix_sparse.dat`)

```
num_edges
a  b  Jr  Ji
a  b  Jr  Ji
...
```

- First line: integer number of undirected edges
- Each subsequent line: `a  b  Jr  Ji` (space-separated), where a < b
- J(a,b) = Jr + i·Ji; J(b,a) = Jr − i·Ji (implicit)
- All values printed with `%0.15f` precision

#### Dense format (`Data_Matrix.dat`)

```
[blank line]
Re(0,0)  Im(0,0)    Re(0,1)  Im(0,1)    ...
Re(1,0)  Im(1,0)    Re(1,1)  Im(1,1)    ...
...
```

- First line is blank (written by legacy code)
- N lines, each with N cells of two space-separated doubles
- Format: `%0.15f  %0.15f    ` (four spaces between cells)
- All N² entries are stored (including zeros and diagonal)
