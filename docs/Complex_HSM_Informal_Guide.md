# Complex_HSM Informal Guide (v2.9)

The *Complex_HSM* program simulates the dynamics of the **Complex Hyperspin Machine**: a generalization of common Ising machines that allows to treat **COMPLEX FIELDS** and **COUPLINGS**, in an **ARBITRARY NUMBER OF DIMENSIONS**, also implementing an **EQUALIZATION SCHEME** to address the problem of amplitude heterogeneity.

This guide is the merger of two documents:

- **Part 1** walks you through what the program *does* — the `Complex_HSM.c` main flow, step by step.
- **Part 2** tells you what is *inside* the supporting library `Complex_HSM.h` — the data structures, the index macros, and every function the `.c` calls on. Specifically what it does, what it takes, what it returns, and (where it matters) the inner algorithm it runs.

If you have not read Part 1 yet, go read it first. Part 2 assumes you know what **N**, **D**, the fields **S**, the equalizers **Y**, and the observables are. Part 2 zooms in on the machinery.

---

# Part 1: The Main Program

---

## 1) Overview and Basic Usage

The program uses libraries:

- `stdio.h`
- `stdlib.h`
- `math.h`
- `string.h`
- `gsl/gsl_eigen.h`, `gsl/gsl_matrix.h`, `gsl/gsl_vector.h`
- `omp.h`

In addition to its own: `Complex_HSM.h`.

The program is compiled as:

```bash
gcc -O3 -fopenmp -march=native src/Complex_HSM.c -o Complex_HSM.x -lgsl -lgslcblas -lm
```

To run the program use:

```bash
./Complex_HSM.x <NS> <pth> <Complex_HSM.ini>
```

- **NS** — the seed used for random generation (of initial conditions and optionally coupling matrices).
- **pth** — the fraction of the pump threshold to use. If `pth >= 0` the pump threshold `hth` is automatically computed and the pump is set to `hm = pth * hth`. Set to `pth = -1` to avoid automatic threshold computation and use the value set in the `.ini` file.
- **Complex_HSM.ini** — the file containing the simulation parameters.

---

## 2) The INI File

The `Complex_HSM.ini` must contain the following parameters as separate lines:

### System Parameters

- `N = <N>` *(int)* — system size
- `D = <D>` *(int)* — hyperspin dimension
- `initial_type = <initial_type>` *(int)*\* — initial condition type
- `Amax = <Amax>` *(double)* — max initial field component amplitude
- `gm = <gm>` *(double)* — intrinsic loss coefficient
- `beta = <beta>` *(double)* — nonlinear interaction coefficient
- `deq = <deq>` *(double)* — equalization coupling strength

### Pump Parameters

- `phim = <phim>` *(double)* — pump average phase
- `dphi = <dphi>` *(double)* — pump phase max deviation
- `hm = <hm>` *(double)* — pump average amplitude
- `dh = <dh>` *(double)* — pump amplitude max deviation
- `Dh = <Dh>` *(double)* — pump dynamic change

### Time Parameters

- `tmax = <tmax>` *(double)* — simulation max time
- `dt = <dt>` *(double)* — time step
- `t_coup = <t_coup>` *(double)* — coupling starting time
- `t_eq = <t_eq>` *(double)* — equalization starting time
- `t_pump = <t_pump>` *(double)* — pump dynamic change time

### Stop Conditions

- `stop_at_E_target = <stop_at_E_target>` *(int)*\*\* — target energy stop condition
- `E_target = <E_target>` *(double)* — target energy value
- `E_target_tol = <E_target_tol>` *(double)* — target energy tolerance
- `stop_when_stable = <stop_when_stable>` *(int)*\*\* — stability stop condition
- `allowed_H_change = <allowed_H_change>` *(double)* — stability allowed change
- `steps_for_update = <steps_for_update>` *(int)* — check energy frequency
- `stable_window = <stable_window>` *(int)* — consequent stability check for stop
- `relative_H_change = <relative_H_change>` *(int)*\*\* — enable relative energy check

### Coupling Generation

- `new_coupling = <new_coupling>` *(int)*\*\*\* — coupling generation type
- `prob_coup = <prob_coup>` *(double)* — generation average connectivity
- `prob_pm_re = <prob_pm_re>` *(double)* — real part positive sign probability
- `prob_pm_im = <prob_pm_im>` *(double)* — imaginary part positive sign probability
- `coup_phase = <coup_phase>` *(double)* — coupling additional global phase
- `type_coup = <type_coup>` *(int)*\* — coupling type
- `lattice_d = <lattice_d>` *(int)* — square lattice generation dimension
- `lattice_NL = <lattice_NL>` *(int)* — square lattice linear size
- `lattice_bc = <lattice_bc>` *(int)*\*\*\* — lattice boundary conditions

### Output Parameters

- `save_dyn = <save_dyn>` *(int)*\*\*\* — output type
- `save_steps = <save_steps>` *(int)* — number of steps between output writes
- `buffer_size = <buffer_size>` *(int)* — buffer size (auto flush when full)
- `flush_interval = <flush_interval>` *(int)* — flush interval (set flushing frequency)
- `output_dir = <output_dir>` *(int)*\*\* — dedicated output directory under `Complex_HSM/results`
- `compute_eigenvalues = <compute_eigenvalues>` *(int)*\*\* — compute eigenvalues at origin & produce dense matrix file
- `num_threads = <num_threads>` *(int)* — number of parallel threads

### Footnotes

> \* — The integer must be **0**, **1**, or **2**: 0 for real numbers, 1 for imaginary numbers, 2 for complex (Hermitian for coupling) numbers.

> \*\* — The integer must be **0** or **1** (Boolean), with 0 for *False* and 1 for *True*.

> \*\*\* — **`new_coupling`**: 0 for reading existing coupling named `Data_Matrix.dat` (dense) or `Data_Matrix_sparse.dat` (sparse); 1 to generate a Random Graph; 2 to generate a Square Lattice. Generating new couplings overwrites old ones. **`lattice_bc`**: use 1 for periodic boundary conditions, 0 for non-connected system boundaries. **`save_dyn`**:
> - `save_dyn = 0` — only saves the fields final state
> - `save_dyn = 1` — saves all the possible observables (see more detail later)
> - `save_dyn = 2` — saves the final state and the Energy and Lyapunov observables
> - `save_dyn = 3` — saves a single file with Energy, Lyapunov, Magnetization and Hyperspin Amplitude Variance

**Note:** Lattice generation parameters are only used when `new_coupling = 2`.

---

## 3) The Main Entry Point

The main takes as arguments:

- **NS** *(int)* — random number generator seed. The program is fully deterministic: using same seed and same parameters implies machine precision reproducibility.
- **pth** *(double)* — if `pth >= 0` the pump threshold `hth` is automatically determined and `hm` is set to `hm = pth * hth`. If `pth = -1` (or `pth < 0` in general) the value of `hm` found in the `Complex_HSM.ini` file is used.
- **Complex_HSM.ini** — the name of the ini file to use to read the simulation parameters.

The program is launched using:

```bash
./Complex_HSM.x <NS> <pth> <Complex_HSM.ini>
```

### Initialization Phase

The main uses the following variable declarations:

- **`int` variables:** `N`, `D`, `save_dyn`, `new_coupling`, `type_coup`, `initial_type`, `stop_at_E_target`, `stop_when_stable`, `steps_for_update`, `stable_window`, `save_steps`, `compute_eigenvalues`, `relative_H_change`, `lattice_d`, `lattice_NL`, `lattice_bc`, `buffer_size`, `flush_interval`, `num_threads`
- **`double` variables:** `Amax`, `phim`, `dphi`, `g`, `beta`, `hm`, `dh`, `Dh`, `deq`, `tmax`, `dt`, `t_coup`, `t_eq`, `t_pump`, `prob_coup`, `prob_pm_re`, `prob_pm_im`, `coup_phase`, `E_target`, `E_target_tol`, `allowed_H_change`, `t_final_value`
- **`char` variables:** `output_dir[512]`, `stop_reason[128]`

**Workflow — function details are given later:**

**Step 0** — If the number of arguments when executing is wrong, return an error message. Relevant variables for initialization are declared.

**Step 1** — `parse_config()` reads the ini file, looks for the simulation parameters and puts each in the corresponding declared variable.

**Step 2** — Relevant input parameters are checked:

- If `save_steps < 1` it is set to 1.
- If `initial_type` is not 0, 1 or 2, return an error message.
- If `output_dir = 1` the output directory is set to `/results`, later the results will be saved under `/results/run_<hash>` where the hash is computed later. If `output_dir = 0` the output directory name is set to `\0` (executable directory).
- If `new_coupling = 2` (Lattice) the ini value of `N` is overridden and set to `N = lattice_NL^d`.
- If `pth > 0` and `compute_eigenvalues = 0` return an error message as `compute_eigenvalues = 1` is required for pump threshold computation.

**Step 3** — `print_header_and_ini_parameters()` prints on screen the header and the main ini parameters.

**Step 4** — The random number generator `srand()` is initialized with `NS`.

**Step 5** — The `system_state` `s` variable is declared (it is a struct type defined in `Complex_HSM.h` — see Part 2). It contains the system state at a given time `t` plus some other useful variables. Memory is allocated according to its size and the program returns a message if the allocation was successful or ends the process on an error message if something went wrong.

**Step 6** — A dummy Lyapunov variable is declared and the `observables` `obs` variable is declared. The observables struct is also defined in `Complex_HSM.h`.

**Step 7** — Fields are initialized according to initial condition type and norms are computed using respectively `initialize_fields()` and `compute_hyperspin_norm()`.

**Step 8** — Coupling matrix is generated or computed according to `new_coupling`. If `compute_eigenvalues = 0` only a sparse version is created (`Data_Matrix_sparse.dat`); if `compute_eigenvalues = 1` a dense version is also built (`Data_Matrix.dat`). An NxN sized matrix is only allocated when the dense matrix is built, while a more optimized amount of memory is used in the other case. Depending on the value of `new_coupling`:

- `new_coupling = 2` → `initialize_J_Hermitian_Lattice()`. Write on file with `write_sparse_matrix_file()`.
- `new_coupling = 1` → `initialize_J_Hermitian_Binary_Sparse()`. Write on file with `write_sparse_matrix_file()`.
- `new_coupling = 0` → `initialize_J_Sparse_From_Sparse_File()` or `initialize_J_Sparse_From_File()`. The program always looks first for a sparse matrix called `Data_Matrix_sparse.dat` and if it doesn't find it looks for `Data_Matrix.dat`. If none is found or there is a mismatch between the system size `N` and the size of the coupling matrix the program will not work properly. If the dense matrix is required (`compute_eigenvalues = 1`) a dense version will be built with `fill_dense_from_sparse()` and saved as `Data_Matrix.dat` with `write_dense_matrix_file()`.

**Step 9** — Apply phase rotation if `coup_phase != 0` to all entries of the coupling matrix with `apply_coupling_rotation()`. If rotation was applied, overwrite the non-phase-rotated data files again with `write_sparse_matrix_file()` and, optionally, `write_dense_matrix_file()`.

**Step 10** — Char variables are declared to store hash unique code and results directory name. Hash code is computed using several system parameters from ini, input line (`NS` and `pth`) and the generated matrix file (sparse format is preferred). The functions `compute_hash()`, `format_hash_hex()`, `copy_file_to_dir()` are used in the process. The output directory name is printed on output.

**Step 11** — If `pth > 0` the pump threshold is computed and the pump value `hm` is set as `hm = pth * hth`. The function `h_threshold()` is used. Notice the function is very naive and not very efficient. This feature should only be used for testing purposes until the code is updated.

**Step 12** — The function `initialize_pump()` is used to set the pump amplitude and phase for each hyperspin, generating them according to a uniform distribution of fixed average `hm` and size.

**Step 13** — Trigonometric functions are precomputed **once** using `precompute_trig()`. Since `phi` is never changed after initialization, this single call covers the entire run — the RK2 core uses the cached `cos_phi`/`sin_phi` and never calls `cos()`/`sin()` again.

**Step 14** — Eigenvalues Computation:

- If `compute_eigenvalues = 1` — first the max eigenvalue of the coupling matrix `J` is computed, then the max eigenvalue of the whole system linearized matrix at the origin is computed (the problem is nonlinear so it is linearized close to the origin). Max eigenvalues are printed on output.
- If `compute_eigenvalues = 0` — max eigenvalues are not computed.

Used functions: `initialize_pump()`, `Linearized_Max_Eigenvalue()`.

**Step 15** — Pump modification array is prepared with `initialize_pump()`.

**Step 16** — Using an `io_context` struct `ctx` (defined in `Complex_HSM.h`) output files are initialized using the function `init_output_files()`. An error message is returned on failure.

**Step 17** — A message that states *'Initialization Completed'* is printed. Also info about peak memory usage is printed on output; peak memory was registered during matrix generation in `initialize_J_Hermitian_Binary_Sparse()`, steady memory is instead directly registered at the calling of `print_memory_peak()`, which also prints the memory information.

**Step 18** — If `num_threads > 1` the number of parallel threads is set with `omp_set_num_threads()`, passing from the default sequential mode with `use_parallel = 0` to parallel with `use_parallel = 1`.

### Dynamics Integration

**Step 19** — Relevant variables for dynamics are declared and initialized. A message is printed that informs the user that the Runge-Kutta integration started.

```c
double t = 0.0;
int steps = 0;
int coup_control = 0;
int eq_control = 0;
int pump_control = 0;
int stable_count = 0;
int first_stable_check = 1;
double H_check_old = 0.0;
int t_final_printed = 0;
```

**Step 20.0** — `while` loop starts for `t < tmax`. Inside the loop:

**Step 20.1** — Time progress is printed on screen as a percentage.

**Step 20.2** — Checks are performed to activate dynamic changes in the system parameters. This affects the coupling matrix (`coup_control`), equalization (`eq_control`) and pump (`pump_control`): if the control variable is zero and the current time is higher or equal to the corresponding dynamic time (`t_coup`, `t_eq`, `t_pump`) the action is performed. *Coup* and *eq* are activated and remain active for the rest of the simulation; pump is modified by the selected `Dh` amount.

**Step 20.3** — If the current time step is a multiple of the `save_steps` variable, a control variable `save_step` is set and **all** the observables are computed and saved on file. Otherwise the `compute_H` gate decides whether any observable is computed at all: on E-target runs (`stop_at_E_target=1`), stability-check-aligned steps (`(steps+1) % steps_for_update == 0`) and the final step, only the Hamiltonian is computed; on all other (idle) steps `compute_observables()` is skipped entirely. The function `compute_observables()` is called with the gate flags `(compute_H, compute_full)`; `save_output_files()` and `flush_output_files()` are only used if `save_step = 1`.

**Step 20.4** — The RK update is performed via the function `RK2_step()`.

**Step 20.5** — Time and steps counter are updated.

**Step 20.6** — If `stop_at_E_target = 1` the program checks if the system's energy is below the selected threshold `E_target`; if so the time is set to `tmax` and the simulation is ended. A message is printed on output with the cause of the stop. The final time is saved and displayed.

**Step 20.7** — If `stop_when_stable = 1` and the current time step is a multiple of `steps_for_update` and the simulation time is higher than `t_coup`, `t_eq` and `t_pump`, the program:

- Saves the current energy if this is the first stability check in the variable `H_check_old`. Otherwise, first checks if the stop energy difference condition must be applied to the relative energy or not, and uses the corresponding value. Then checks whether the condition is satisfied and if so updates the counter `stable_count` adding one, otherwise it resets it to zero.
- If `stable_count > stable_window` time is set to `tmax` and the simulation is ended. A message is printed on output with the cause of the stop. The final time is saved and displayed.
- If the program did not stop, updates the variable `H_check_old` with the current value of the energy.

### Finishing Steps

**Step 21** — After the loop, if the finishing time was not printed (the stop conditions also update a corresponding control variable) the program prints that the simulation has reached the maximum time.

**Step 22** — The final configuration of the system dynamics variables is saved on file using the function `save_final_config()`.

**Step 23** — If `output_dir = 1` (hence `results_dir[0]` is not an empty string but contains the name of the results directory) a metadata file is written in it with the `write_metadata()` function. If `output_dir = 0` (hence `results_dir[0]` is an empty string) this step is skipped.

**Step 24** — If `save_dyn > 0` output files that were open are closed with the function `close_output_files()`; else only the final configuration file is closed.

**Step 25** — The final value of the energy is printed on screen.

**Step 26** — All the dynamics variables of the system are freed with the function `free_system()`.

---

# Part 2: The Library

---

## 4) What Kind of File Is This?

`Complex_HSM.h` is a single big header that the main program `Complex_HSM.c` `#include`s.

It is **NOT** a "header only declares, the `.c` implements" file. Everything in here (plain inline helper, struct, macro, or full function body) lives directly in the header.

Consequences you should absorb:

- There is **no separate** `Complex_HSM_lib.c` to compile. You compile **ONLY** `Complex_HSM.c` and that one file drags in all this code. (The compiler sees one big translation unit.)
- Because everything is in one file, internal helper functions do not need to be "declared before use" the way they must across `.c` files — the `.h` just orders things sensibly on its own.

The header also pulls in a stack of system headers on top of what the `.c` already includes:

- `sys/stat.h`, `sys/types.h` — `mkdir()` and friends (creating the results directory)
- `unistd.h` — `access()` (checking if a file exists)
- `time.h` — `time()`, `localtime()`, `strftime()` (for metadata timestamps)
- `sys/resource.h` — `getrusage()` (RSS memory measurement)
- `malloc.h` — `mallinfo2()` (heap memory measurement)

You do not need to see these includes again; just know that every library routine below that needs the filesystem, the clock, or memory stats has its headers right here.

---

## 5) Data Structures

---

### 5.1) `system_state`

This is the main system variable. It holds everywhere the simulation is at a given time `t`, plus some precomputed scratch and bookkeeping. The `.c` declares exactly one of these (`s`), allocates its arrays, and passes `&s` (a pointer to it) to almost every library function. That is why you see `system_state *s` in nearly every signature.

**The single most important design idea:** ALL the big arrays are **FLAT 1D arrays**, not arrays of pointers / arrays of structs. There are no pointer chains. Field `(a, component j)` lives at index `a*D + j` in a contiguous block. This choice was made because contiguous memory is cache-friendly and lets OpenMP threads write disjoint slices without false sharing.

Here is the full tour (sizes given in parentheses; **N** = number of hyperspins, **D** = number of complex components per hyperspin):

#### Fields

- `S_re`, `S_im` *(N*D each)* — hyperspin fields, real and imaginary parts. These are the variables of the physical dynamic system. A single field is a complex number given by `S(i,j) = S_re[i*D+j] + i*S_im[i*D+j]`: `i` and `j` represent the hyperspin number and the specific component (D of them).
- `S2` *(N)* — the squared norm of each field, `S2(a) = |S(a)|^2 = sum_j (Re^2 + Im^2)`. Recompute via `compute_hyperspin_norm()`; used everywhere (normalization, beta nonlinearity, energy).
- `DS_re`, `DS_im` *(N*D each)* — the **MIDPOINT** fields. RK2 is a two-stage integrator: it first takes a half-step to produce DS (the midpoint), then a full step from the midpoint to update S. DS lives here. Not meaningful between the two phases — it is transient.
- `DS2` *(N)* — squared norm of the midpoint fields, same idea as `S2` but for DS.

#### Equalizers (Y fields)

The equalization scheme that fights amplitude heterogeneity:

- `Y_re`, `Y_im` *(N\*2 each)* — each hyperspin carries **TWO** complex equalizer fields, indexed by `k = 0,1`. Equalizer `(a,k)` = `Y_re[(a)*2 + k] + i*Y_im[(a)*2 + k]`.
- `DY_re`, `DY_im` *(N\*2 each)* — the midpoint equalizers, same role as DS for the fields. The equalizers are integrated with the same RK2 midpoint/full scheme. Note the size is `N*2` as equalizers are always **2 per hyperspin**, regardless of D. (The header indeed prints `Neq=2*N`.)

#### Pump and Phase

This is how the injected "pump" is stored, per hyperspin:

- `h` *(N)* — pump amplitude per hyperspin, drawn around the average `hm` with spread `dh`.
- `hmod` *(N)* — the pump **MODIFICATION** array, same structure as `h`. Filled once with the value `Dh`; it can for instance be used with `modify_pump()` to modify `h[a]` shifting it by a constant vector.
- `phi` *(N)* — pump phase per hyperspin, drawn around average `phim` with spread `dphi`.

#### Optimization

- `cos_phi`, `sin_phi` *(N each)* — `cos(phi[a])` and `sin(phi[a])` precomputed once at startup by `precompute_trig()` (since `phi` is constant during the run), so the hot RK2 loop never calls `cos()`/`sin()`.
- `inv_sqrtS2` *(N)* — reusable scratch holding `1/sqrt(S2(a))`; computed in `compute_observables` whenever `compute_H=1` and shared by the H and magnetization passes.

#### Sparse Coupling

The coupling matrix `J`, stored sparsely — built **ONCE**, used in hot loops:

- `csr_row_ptr` *(N+1)*, `csr_col_idx` *(nnz)*, `csr_Jr` *(nnz)*, `csr_Ji` *(nnz)* — the coupling matrix in **Compressed Sparse Row** format. `row_ptr[a]..row_ptr[a+1]` is the run of non-zeros in row `a`; each entry has a column index and real/imag values.
- `num_edges` — the number of **UNDIRECTED** edges in the coupling graph (`nnz/2`, since J is Hermitian and each off-diagonal pair is stored twice, once per triangle).
- `edge_a`, `edge_b` *(num\_edges each)* — the two endpoints of each undirected edge. By construction `edge_a[e] < edge_b[e]` (smaller index first).
- `edge_Jr`, `edge_Ji` *(num\_edges each)* — the coupling value along that edge, stored as the **UPPER-triangle** value `J(edge_a, edge_b)`. The lower triangle is its Hermitian conjugate.

#### Dense Coupling

Only present when needed:

- `J_re`, `J_im` *(N\*N each)* — the dense coupling matrix. These are `NULL` until *compute eigenvalues* is enabled. Only allocated/filled when `compute_eigenvalues=1` (needed for `h_threshold` and the `Jmax`/`Lmax` eigenvalue diagnostics, which need a real NxN matrix).
- `J_allocated` *(int)* — **1** if `J_re`/`J_im` were actually allocated. Guards `free_system()` and every place that touches the dense matrix so we do not dereference `NULL`. In sparse-only mode this stays **0** and the arrays stay `NULL`, saving O(N²) memory.

#### Memory Accounting

For the memory-use reports:

- `startup_heap_B`, `steady_heap_B` *(unsigned long)* — peak heap after matrix generation and steady-state heap after I/O buffers, measured via `mallinfo2()`. Reported by `print_memory_peak()`.

---

### 5.2) `observables`

A small struct that bundles **ALL** the diagnostics so `compute_observables()` can return them in one shot:

- `H` *(double)* — the Hamiltonian / energy. Only computed when `compute_H=1` (save-steps multiples, E-target runs, stability-check-aligned steps, final step); otherwise it is **0** and the `.c` skips `compute_observables()` entirely (v2.9 `compute_H` gate). The stability check reads the `H` computed one step earlier — always valid by construction.
- `L`, `L0`, `Lc` *(double)* — the Lyapunov function `L` and its two pieces: `L0` (not interacting part) and `Lc` (coupling / interaction). Only computed when `compute_full=1`.
- `dL` *(double)* — `(L - L_old)/dt`, the time-derivative of L. Only when `compute_full=1`.
- `m` *(double)* — average magnetization. Only when `compute_full=1`.
- `DS2` *(double)* — the average variance of the field amplitudes (mean of `|S|²` minus the square of its mean, `/N`). Only when `compute_full=1`.

The split *"H only when `compute_H=1`, the rest only on `compute_full=1`"* is the whole point of the `save_steps` + `compute_H` optimizations (see Part 1, step 20.3, and the `compute_observables()` details in section 7.6).

---

### 5.3) `io_context`

A bundle of `FILE*` handles for the buffered output. Created by `init_output_files()` and passed to every output writer. Its fields are one handle per possible output file:

- `f_fields` — `Data_Fields.dat` *(save_dyn=1)*
- `f_equalizers` — `Data_Equalizers.dat` *(save_dyn=1)*
- `f_hamilton` — `Data_Hamilton.dat` *(save_dyn=1 or 2)*
- `f_lyapunov` — `Data_Lyapunov.dat` *(save_dyn=1 or 2)*
- `f_h_ly_m_ds2` — `Data_H_Ly_m_DS2.dat` *(save_dyn=3)*
- `f_final` — `Data_Final.dat` *(ALL modes)*

Only the handles for the active `save_dyn` mode are non-`NULL` (plus `f_final` always). The writers check `save_dyn` to decide which handles to touch.

---

### 5.4) `use_parallel` and `print_memory_peak()`

- `int use_parallel = 0;` is a **GLOBAL** `int` living in this header. It is the runtime gate for all OpenMP. The `.c` sets it to **1** when `num_threads > 1` and sets the thread count. Every `#pragma omp parallel for ... if(use_parallel)` in this file checks it: when **0**, the pragma is a no-op (no thread team, no overhead) and a fully sequential computation is implemented; when **1**, parallel threads are used.
- `print_memory_peak(system_state *s)` is a static inline helper (negligible overhead, called once). It reads `getrusage()` for peak RSS and `mallinfo2()` for the heap, stores `steady_heap_B` into `s`, and prints all three. It is how the `.c` reports 'startup' and 'steady' heap numbers.

---

## 6) The Index Macros

The flat arrays are indexed through a set of macros. They encapsulate the row-major layout so a future layout change does not ripple through the whole code.

### Hyperspin Component Access

`(a = spin 0..N-1, j = component 0..D-1)`

- `S_RE(s, a, j)` → `s->S_re[(a)*D + j]`
- `S_IM(s, a, j)` → `s->S_im[(a)*D + j]`
- `S2(s, a)` → `s->S2[a]`

### Midpoint (DS) Access

Same shape:

- `DS_RE(s, a, j)`, `DS_IM(s, a, j)` → `DS_re/DS_im[(a)*D + j]`
- `DS2(s, a)` → `DS2[a]`

### Equalizer Access

`(a = spin, k = 0 or 1)` — note the stride is **2**, not D for them:

- `Y_RE(s, a, k)`, `Y_IM(s, a, k)` → `Y_re/Y_im[(a)*2 + k]`
- `DY_RE(s, a, k)`, `DY_IM(s, a, k)` → `DY_re/DY_im[(a)*2 + k]`

### Coupling Matrix Access

Only meaningful when the dense J is allocated:

- `J_RE(s, a, b)` → `J_re[(a)*N + b]`
- `J_IM(s, a, b)` → `J_im[(a)*N + b]`

### Precomputed Coupling Product

The J\*S result:

- `JS_RE(s, a, j)`, `JS_IM(s, a, j)` → `JS_re/JS_im[(a)*D + j]`

The pattern `(a)*stride + index` is the same everywhere; only the stride changes (**D** for fields, **2** for equalizers, **N** for the dense coupling matrix).

---

## 7) Function Families (The Reference)

Below, the functions are grouped the same way the file groups them. Each one is described in plain words, the way Part 1 talks: first what it does and when the `.c` calls it, then, where it matters, a walkthrough of the inner steps. Function calls are written `name()`; when a function below uses another function from this guide, the name appears right in the text so you can jump around.

---

### 7.1) Input Feedback and Header Printing

---

**`void print_header_and_ini_parameters(filename, N, D, initial_type, NS, Amax, phim, dphi, g, beta, hm, dh, Dh, pth, deq, tmax, dt, t_coup, t_eq, t_pump, save_dyn, new_coupling, prob_coup, prob_pm_re, prob_pm_im, coup_phase, type_coup, E_target, E_target_tol, stop_at_E_target, stop_when_stable, allowed_H_change, steps_for_update, stable_window, relative_H_change, buffer_size, flush_interval, num_threads, save_steps, compute_eigenvalues, lattice_d, lattice_NL, lattice_bc)`**

Prints the nice ASCII-art banner and then echoes every simulation parameter in nicely sectioned blocks (system, cavity/equalization, coupling, time, pump threshold, end conditions, output files, buffered I/O, parallelization). It is purely cosmetic: it returns nothing and changes nothing, and its giant signature mirrors exactly the variables the `.c` parsed from the ini. One ordering detail worth keeping straight: the `.c` calls it **BEFORE** `srand()` is seeded, but **AFTER** the lattice override has already happened — step 2 of the `.c` workflow replaces `N` with `NL^d` before this function prints — so in lattice mode the banner already shows the final `N` that will actually be run.

---

**`int parse_config(filename, &N, &D, &initial_type, &Amax, &phim, &dphi, &g, &beta, &hm, &dh, &Dh, &deq, &tmax, &dt, &t_coup, &t_eq, &t_pump, &save_dyn, &new_coupling, &prob_coup, &prob_pm_re, &prob_pm_im, &coup_phase, &type_coup, &E_target, &E_target_tol, &stop_at_E_target, &stop_when_stable, &allowed_H_change, &steps_for_update, &stable_window, &relative_H_change, &buffer_size, &flush_interval, &num_threads, &save_steps, output_dir, output_dir_sz, &compute_eigenvalues, &lattice_d, &lattice_NL, &lattice_bc)`**

Opens the ini file, reads every `key = value` line, and fills the corresponding variable through a pointer; it returns **0** on success and **1** if the file cannot be opened. It starts by loading the default for every parameter that has one:

- `buffer_size` = 262144
- `flush_interval` = 1000
- `num_threads` = 1
- `stable_window` = 1000
- `relative_H_change` = 0
- `steps_for_update` = 1
- `E_target_tol` = 1e-10
- `save_steps` = 1
- `compute_eigenvalues` = 1
- `lattice_d` = 2, `lattice_NL` = 10, `lattice_bc` = 1
- `initial_type` = 2
- `prob_pm_re/im` = 0.5, `coup_phase` = 0

That is why the lattice parameters can be missing when you are running a random graph or an input matrix — the program just keeps the default.

Then it loops over the lines one by one. Lines starting with `#` and blank lines are skipped (that is how comments work in the ini). Each line is split at the `=` into key and value, both trimmed of surrounding spaces/tabs/CR/LF, and an if/else chain matches the key string to one of the known names and parses the value (`atoi` for ints, `strtod` for doubles, `strncpy` for the output_dir string). Remember the special key `gm`: the ini file calls it `gm`, the program calls it `g` — `gm` maps onto the variable `*g`. Unknown keys are reported to stderr as *'Unknown key:'* but are **NOT** fatal; the rest of the file keeps parsing. No range check happens here — the sanity checks (`initial_type`, `save_steps`, `pth` vs `compute_eigenvalues`) are done later in the `.c`, exactly where they are needed.

---

### 7.2) System Memory Management

---

**`int allocate_system(int N, int D, system_state *s)`**

Allocates every array of the `system_state` and initializes the scalar bookkeeping; it returns **0** on success and **1** if any allocation failed. It sets `s->N` and `s->D`, then `calloc()`s:

- `S`/`DS` *(N\*D)*, `S2`/`DS2` *(N)*, `Y`/`DY` *(N\*2)*
- `h`/`hmod`/`phi` *(N)*, `JS` *(N\*D)*
- `cos_phi`/`sin_phi`/`inv_sqrtS2` *(N)*

`J_re` and `J_im` are set to `NULL` with `J_allocated = 0`, because the dense coupling matrix is allocated later and only if `compute_eigenvalues = 1`. Every pointer is then checked; any `NULL` returns **1**. The sparse arrays (`csr_*`, `edge_*`) are **NOT** allocated here either — the coupling-construction functions do that. Since everything is `calloc`'d, all arrays start clean (all zeroes).

---

**`void free_system(system_state *s)`**

Releases everything `allocate_system` and the coupling builders allocated. It frees each array, but frees `J_re` and `J_im` **ONLY** when `J_allocated` is set — that single test avoids both double-freeing and dereferencing `NULL`. It also frees the sparse and edge arrays. It ends with `memset(s, 0)`, so the whole struct is left all-zero afterwards.

---

### 7.3) Field / Pump / Trig Initialization

---

**`void initialize_fields(system_state *s, double Amax, int initial_type)`**

Draws the initial field and equalizer values uniformly in `[-Amax, Amax]` and then applies the `initial_type` constraint. For every spin `i` and component `j` it **ALWAYS** draws both a random real part and a random imaginary part (two `rand()` calls), and only THEN:

- If `initial_type==0` — zeroes the imaginary part.
- If `initial_type==1` — zeroes the real part.

It does exactly the same for the equalizers. It never skips a draw, and **WHY that matters**: the global `rand()` stream is shared with the coupling-matrix generators, so if this function were "optimized" by skipping the unused draw, it would change how many `rand()` calls happen before the matrix generation, and the **SAME** seed would produce a **DIFFERENT** coupling matrix. That is why `initial_type=2` is byte-for-byte identical to older versions, and why changing `initial_type` does **NOT** change the matrix. The `.c` validates `initial_type` before calling, so this function can assume it is already 0, 1 or 2.

---

**`void initialize_pump(system_state *s, double *target, double hm, double dh)`**

Fills an arbitrary N-array `target` with pump values drawn uniformly on `[hm - dh, hm + dh]`. For each `i`: `a = -dh + 2*dh*(rand()/(RAND_MAX+1))`; `target[i] = hm + a`. That is a uniform spread of width `2*dh` centered on `hm`. Because `target` is just a parameter, the **SAME** function serves three different jobs (all visible in the `.c` init):

- `initialize_pump(&s, s.h, hm, dh)` → the actual pump amplitudes, around `hm`
- `initialize_pump(&s, s.phi, phim, dphi)` → the pump phases, around `phim`
- `initialize_pump(&s, s.hmod, Dh, 0)` → the pump-modification array, with spread 0, so every entry is exactly `Dh` and `modify_pump()` later shifts every `h` by the same uniform step `Dh`

---

**`void modify_pump(system_state *s)`**

Applies the pump modification: it adds `hmod[i]` onto `h[i]` for every `i`. `s->h[i] += s->hmod[i]`, that is all. The `.c` calls it exactly once, at `t_pump`.

---

**`void precompute_trig(system_state *s)`**

Computes `cos(phi[i])` and `sin(phi[i])` for every `i` into the scratch arrays `cos_phi`/`sin_phi`: a simple loop of the two transcendentals per spin. It is called **once at startup** (right after `initialize_pump`, before the dynamics loop), so the hot RK2 core never calls the slow `cos()`/`sin()` itself — `phi` never changes during the run.

---

### 7.4) Coupling Matrix Construction

The biggest, subtlest family. There are generically **FOUR** ways to get the coupling matrix, selected by `new_coupling` in the `.c` program, plus several helpers they all share. Before the functions, two conventions to keep in your head.

#### Hermitian Convention

- `J` is Hermitian: `J(b,a) = conjugate(J(a,b))`. The code only ever stores or creates the **UPPER triangle**, meaning `J(a,b)` for `a<b`, and derives the lower triangle by conjugation.
- A single coupling `(a,b)` has a real part and an imaginary part. In the code the upper-triangle value is written as `J = J1 + i*J2`.
- The **EDGE list** stores that upper-triangle value as `(edge_a, edge_b)` with `edge_a < edge_b`, and its `Jr + i*Ji` is exactly `J(edge_a, edge_b)`. So reading an edge gives you the upper triangle directly.

#### Normalization

Coupling magnitudes are scaled by the expected degree so the total coupling strength does not blow up with `N`. Depending on `type_coup`:

- `type_coup 0` (real only): `dJR = 1/(N*prob_coup)`, `dJI = 0`
- `type_coup 1` (imag only): `dJR = 0`, `dJI = 1/(N*prob_coup)`
- `type_coup 2` (both): `dJR = dJI = 1/(sqrt(2)*N*prob_coup)`, the `sqrt(2)` keeping `|J| = 1/(N*prob_coup)`, the same strength as the single-component cases

For the lattice, `N*prob_coup` is replaced by `degree*prob_coup` where `degree = 2*lattice_d`. `prob_coup` appears because it is the expected number of advanced-neighbour pairings: in the random case that means `prob_coup*n(n-1)/2` edges, an average degree of about `N*prob_coup`.

---

**`void initialize_J_Hermitian_Binary_Random(system_state *s, prob_pm_re, prob_pm_im, prob_coup, type_coup)`**

The **LEGACY** dense random-graph generator: it builds the full NxN `J` in the dense arrays and writes `Data_Matrix.dat`. It is the old version, kept for reference, and it is the one whose matrix layout the dense file format matches. It zeroes the diagonal, then loops over each pair `(i,j)` with `j < i` and flips an existence coin: the edge is present with probability `prob_coup`. If present, it draws two independent signs (real sign from `prob_pm_re`, imaginary sign from `prob_pm_im`), sets `J1 = dJR*sign_re` and `J2 = -dJI*sign_im`, and writes the LOWER-triangle element `(i,j)` as `(J1, -J2)` and the UPPER-triangle element `(j,i)` as `(J1, +J2)`, exactly what Hermiticity requires (lower triangle = conjugate of upper). It uses the system `rand()` and writes `Data_Matrix.dat` in the dense two-numbers-per-element row format.

*Watch out:* this is a dense O(N²) function, and memory-hungry for that reason. The current `.c` does **NOT** call it for generation — the real random-graph path is the sparse generator below.

---

**`void initialize_J_Hermitian_Binary_Sparse(system_state *s, prob_pm_re, prob_pm_im, prob_coup, type_coup, int NS)`**

**THE** sparse-first random-graph generator. It builds the CSR and edge arrays **DIRECTLY** (no dense J is ever allocated), writes `Data_Matrix_sparse.dat`, and keeps the whole construction deterministic in the seed `NS` alone. Memory is O(N + nnz): the O(N²) decision array of a naive generator is avoided by re-computing decisions instead of remembering them. The whole trick is **TWO IDENTICAL PASSES**, both driven by a **LOCAL** `rand_r(&seed)` re-seeded with `NS` — never the global `rand()` stream.

The normalization first, following the family convention (expected degree ~N\*prob\_coup):

- `type_coup 0` → `dJR = 1/(N*prob_coup)`, `dJI = 0`
- `type_coup 1` → `dJR = 0`, `dJI = 1/(N*prob_coup)`
- `type_coup 2` → `dJR = dJI = 1/(sqrt(2)*N*prob_coup)`

**Pass 1 (count)** seeds `rand_r` from `NS` and loops over every pair `(i,j)` with `j < i`. Each pair draws ONE random number: the edge exists if `rand_r()/RAND_MAX < prob_coup`. If it does, two MORE draws are taken and **DISCARDED** (burned, not stored), and `row_count[i]++`, `row_count[j]++`, `nnz += 2` — the upper-triangle entry in row `i` plus the lower-triangle conjugate in row `j`. Burning the two sign draws is the key trick: Pass 2 also consumes them, so both passes see the **IDENTICAL** `rand_r` sequence and reach exactly the same edge decisions. Then `num_edges = nnz/2` (two CSR entries per undirected edge), the arrays are allocated (`csr_row_ptr` of size N+1, `csr_col_idx`/`csr_Jr`/`csr_Ji` of size nnz, `edge_a`/`edge_b`/`edge_Jr`/`edge_Ji` of size num_edges), and `csr_row_ptr` is built as the prefix sums of `row_count`: `csr_row_ptr[i+1] = csr_row_ptr[i] + row_count[i]`.

**Pass 2 (fill)** re-seeds `rand_r` from `NS` **AGAIN** and re-runs the identical pair loop, so every edge decision comes back exactly. For each existing edge the two sign draws are now **USED**: `sign_re = (rand_r/RAND_MAX < prob_pm_re)` and `sign_im` similarly, each turned into +/- 1 via `coup_sign = -1 + 2*sign`, giving `J1 = dJR*coup_sign_re` and `J2 = -dJI*coup_sign_im`. The coupling is then written in three consistent places. Since `j < i`, the pair `(j,i)` is the upper triangle:

- CSR row `i` → col `j`: `(J1, -J2)`, i.e. `J(i,j) = J1 - i*J2` (the **lower** entry)
- CSR row `j` → col `i`: `(J1, +J2)`, i.e. `J(j,i) = J1 + i*J2` (the **upper** entry)
- Edge list: `edge_a=j`, `edge_b=i` (smaller index first), carrying `(J1, +J2)` — exactly `J(edge_a, edge_b)`

The two CSR entries are complex conjugates, so J stays Hermitian. Finally it writes `Data_Matrix_sparse.dat` through `write_sparse_matrix_file()`, frees the scratch arrays `row_count` and `cur_row`, and records `startup_heap_B` via `mallinfo2()` for the memory report.

> **WHY TWO IDENTICAL RANDOM PASSES** *(keep this idea close)*: so nothing has to be remembered. A naive generator would store the full NxN `pair_data` decision array (O(N²) memory); here Pass 2 recomputes every decision for free by re-seeding `rand_r` from `NS`. Because `NS` is the `rand_r` seed itself, the matrix is fully deterministic in `NS` — and independent of the global `rand()` stream, so it never perturbs the field/equalizer initialization the `.c` drew earlier from `srand(NS)`.

---

**`void initialize_J_Hermitian_Lattice(system_state *s, lattice_d, lattice_NL, lattice_bc, prob_pm_re, prob_pm_im, prob_coup, type_coup)`**

Builds a d-dimensional hypercubic lattice of couplings, `NL` per side, so `N = NL^d`. The Hermitian + normalization structure is the same as the random case, but the degree is `2*d` instead of about `N*prob_coup` (see the Normalization section above), and it handles periodic (PBC) versus open (OBC) boundary conditions. First it validates that the `N` the `.c` gave equals `NL^d`; if not, it prints an error and returns. It uses two small coordinate helpers (`index_to_coord` / `coord_to_index` — described right below) that map a flat spin index to a d-tuple and back, row-major with the last dimension fastest.

**Pass 1 (decide)** runs, for each spin `a` and each dimension, in both directions. The forward neighbor is `coord[dim]+1`, the backward is `coord[dim]-1`. PBC wraps the coordinate (`>= NL` goes to 0, `< 0` goes to `NL-1`); OBC marks the direction invalid instead. Whenever a valid neighbor `b > a` is found (that is the upper triangle), it draws existence from `prob_coup` and, if present, the two signs from `prob_pm_re`/`prob_pm_im`, bumping the row counts and `total_edges`. Crucially, **ALL** decisions — even the invalid and the non-existing ones — are appended to a packed `edge_decisions` array, so Pass 2 reproduces them without re-drawing. That is what keeps the lattice generator deterministic in the global `rand()` stream. It then builds the CSR prefix sum, allocates the CSR + edge arrays, and zeroes the dense J if it was allocated (`J_allocated`).

**Pass 2 (fill)** walks the same loops, uses the stored decisions to fill the CSR rows (with the same upper/lower Hermitian mirroring as before) and the edge list, and fills the dense J if present. As in the sparse generator, the stored edge value is `(dJR*sign_re, -dJI*sign_im)` — the upper-triangle `J(edge_a, edge_b)`. It finishes by writing the sparse file, and the dense file if the dense J exists.

> **NOTE** the lattice generator **DOES** call the global `rand()` (not `rand_r`), so it is coupled to the main `rand()` stream started by `srand(NS)` in the `.c` — that is fine, it is deterministic in `NS` too, just through the shared stream. *(Contrast with the sparse random generator, which shields itself with `rand_r`.)*

Here are the two coordinate helpers (the lattice's address book; you rarely need them outside lattice mode):

- `static inline void index_to_coord(int idx, int NL, int d, int *coord)` — folds a flat index into d coordinates via repeated `% NL` and `/ NL` (last dimension fastest), filling the passed int array.
- `static inline int coord_to_index(const int *coord, int NL, int d)` — the inverse: accumulates `coord[dim] * stride` with `stride *= NL` each dimension, returning the flat index.

---

**`void initialize_J_Read_From_File(system_state *s, const char *filename)`**

Reads a coupling matrix from a DENSE-format file (`Data_Matrix.dat`: a raw N × N grid, two numbers per element, real and imaginary) straight INTO the dense arrays `J_re`/`J_im`; the sparse structures are **NOT** touched. It is a plain double `fopen`/`fscanf` loop reading `%lf %lf` into `J_RE(s,i,j)` and `J_IM(s,i,j)` for every element — no counting, no two-pass trick.

*Watch out:* this is the OLD v8-era reader and is **NOT** called by the current `Complex_HSM.c` — the `.c` always goes through `initialize_J_Sparse_From_(Sparse_)File` below. It is kept as a legacy utility; its sibling `initialize_J_Sparse_From_File` is the one that turns a dense-format file into the sparse structures the current code actually uses.

---

**`void initialize_J_Sparse_From_File(system_state *s, const char *filename)`**

Reads a coupling matrix from a DENSE-format file (the raw N × N grid, two numbers per element) and builds the sparse CSR + edge structures from it: the backwards-compatible reader for old dense files. It works in two passes. Pass 1 scans every element, counts the non-zero off-diagonals per row to size the CSR arrays, sets `num_edges = nnz/2` and allocates. Pass 2 rewinds and re-reads, filling the CSR and the edge list (for `i<j` only, the upper triangle). It assumes Hermitian structure — it only records the upper triangle and lets the CSR carry both. It needs the file to actually exist and match `N`; the `.c` only falls back here when the sparse file is absent.

---

**`void initialize_J_Sparse_From_Sparse_File(system_state *s, const char *filename)`**

Reads a coupling matrix from a SPARSE-format file (`Data_Matrix_sparse.dat`: a header line with the edge count, then one edge `a b Jr Ji` per line, `a < b`, with `Jr + i*Ji` the upper-triangle value) and reconstructs the CSR and edge structures. Pass 1 reads the edge-count header, then each edge, bumping `row_count` for both endpoints (a full row count = the degree) to size the CSR; total `nnz = sum of the row counts`. It allocates the CSR + edge arrays. Pass 2 rewinds, skips the header, re-reads each edge and fills: CSR row `a` → col `b` with `(Jr, Ji)` [upper], CSR row `b` → col `a` with `(Jr, -Ji)` [lower, the Hermitian conjugate], and the edge list as-read (`a, b, Jr, Ji`). This is the canonical reader for the self-generated sparse files.

---

**`void build_sparse_coupling(system_state *s)`**

Rebuilds the sparse CSR + edge structures **FROM** the dense arrays — the reverse direction of `fill_dense_from_sparse` — and is used when a dense matrix already exists. Pass 1 scans the dense J, counts the non-zero off-diagonals per row and sizes everything; Pass 2 scans again and fills the CSR and the edge list (for `a<b`). Essentially the same two-pass shape as the file readers, but the source is the in-memory dense J.

---

**`void fill_dense_from_sparse(system_state *s)`**

Fills the dense J arrays from the sparse edge list (sparse → dense). It is used when `compute_eigenvalues = 1` but the matrix came from sparse generation or from a file: the dense copy must be built for the eigenvalue routines, and `Data_Matrix.dat` is written from it. It zeroes both dense arrays, then for each edge `(a,b)` with values `(Jr,Ji)` sets `J_RE(a,b)=Jr`, `J_IM(a,b)=Ji`, `J_RE(b,a)=Jr` and `J_IM(b,a)=-Ji` — restoring Hermiticity into the dense form, since the edge is upper-triangle-centric and the lower entry gets the conjugate.

---

**`void write_sparse_matrix_file(system_state *s, const char *filename)`**

Writes the sparse edge list to a file: the header is `num_edges`, then one line per edge `a  b  Jr  Ji`. Every generator calls it. It requires the edge list to exist — it prints an error and returns if it does not.

---

**`void write_dense_matrix_file(system_state *s, const char *filename)`**

Writes the dense J to a file (`Data_Matrix.dat`): a blank first line, then N rows, each with N pairs of numbers (Re then Im) separated by extra spaces. It is only callable when the dense J is allocated.

---

**`void apply_coupling_rotation(system_state *s, double coup_phase)`**

Multiplies every coupling entry by the global phase `e^(i*coup_phase)` in place; this is the v2.7 `coup_phase` feature. If `coup_phase == 0` it returns immediately (a no-op). Otherwise it precomputes `rc = cos(phase)`, `rs = sin(phase)`, and for each edge turns the old value `(oldJr, oldJi)` into `(oldJr*rc - oldJi*rs, oldJr*rs + oldJi*rc)` — a rotation of the complex number by that angle. It then writes the rotated value into the edge list, into the CSR upper entry (row `a` → col `b`), into the CSR lower entry (row `b` → col `a`, as the conjugate `-newJi`), and into the dense J if allocated (again with the conjugate on the lower). Because it touches edge + CSR + (optionally) dense, all three representations stay consistent after the rotation. The `.c` then re-writes the matrix files to persist it. It should only be run **AFTER** the matrix is fully built — which is exactly where the `.c` calls it. *(This is a historical feature label; the v2.9 description is otherwise identical.)*

---

### 7.5) The Update Functions (The Hot Loop)

---

**`double S2_of_spin(system_state *s, int a)`**

Returns the squared norm `|S(a)|^2 = sum_j (Re(S)^2 + Im(S)^2)` for one spin. A pure helper: it uses `s->D` so it loops over exactly the components, and the sum of squares is always `>= 0`. `compute_hyperspin_norm()` calls it for all spins.

---

**`double Y2eq_of_spin(system_state *s, int a)`**

Returns the equalization "signed norm" of hyperspin `a`: `( |Y(a,0)|^2 - |Y(a,1)|^2 )`. This difference is what drives the equalizer dynamics — `Y0` 'wants' spin `a`, `Y1` 'wants' its neighbor, and the sign tells the direction. Note it **CAN** be negative, unlike `S2`, because it is a difference.

---

**`void compute_hyperspin_norm(system_state *s)`**

Recomputes `s->S2[i] = S2_of_spin(s, i)` for all spins. It is called at initialization and at the end of every `RK2_step`, so the `S2` array is always current for the next step.

---

**`void compute_coupling_product_flat(const system_state *s, const double *src_re, const double *src_im, double *out_re, double *out_im)`**

Computes the sparse matrix-vector-like product `JS = J * src`: for every spin `a` and component `j`, `out[aD+j] = sum_b J(a,b) * src[bD+j]`. This is **THE** coupling matvec used in both halves of the RK2 step. It is called with either the current fields (`src = S`) for the midpoint, or the midpoint fields (`src = DS`) for the full update; `out` is always `JS`.

Over the outer spin loop `a` it zeroes `out[a]`, then walks row `a` of the CSR (the entries `k` in `row_ptr[a]..row_ptr[a+1]`); for each neighbor `b` with coupling `(Jr,Ji)` it accumulates over the components `j`:

- `out_re += Jr*src_re - Ji*src_im`
- `out_im += Jr*src_im + Ji*src_re`

In words it applies the complex multiply `J*src`: real = `Jr*re - Ji*im`, imag = `Jr*im + Ji*re`. Because it uses the CSR and not the dense matrix, the cost is proportional to the number of edges, not N² — this is the function that makes the whole sparse-first design pay off. Under OpenMP it runs with `#pragma omp parallel for schedule(static) if(use_parallel)` over `a`, and each thread writes a disjoint slice of `out`, so there are no races.

---

**`void RK2_step(system_state *s, double dt, double g, double beta, double deq, int coup_control, int eq_control)`**

Advances the whole system by one time step `dt` using a second-order Runge-Kutta (the midpoint method). It updates `S`, `DS` (transiently), `Y`, `DY`, and refreshes `S2`. Two control flags turn the coupling term (`coup_control`) and the equalization term (`eq_control`) on or off — they are **0** until the corresponding activation time, then **1** for the rest of the run.

**MIDPOINT PHASE:** it first computes `J*S` once into `JS` (`compute_coupling_product_flat` with `src=S`). Then, for each spin `a` in parallel, it finds the ring neighbors `left = (a+N-1)%N` and `next = (a+1)%N` — **NOTE** the equalization topology is a **RING**: spin `a`'s equalizers talk to its immediate left and right cyclic neighbors, **NOT** the coupling lattice; coupling and equalization are independent graphs. It pulls the precomputed `cos_phi[a]`/`sin_phi[a]` from the trig scratch, computes spin-independent coefficients once per spin from the NO-current couplings (the `f0` real/imag pieces involving `h`, `g`, `beta*S2`, and the cross term from `phi`, each halved by `0.5` for the midpoint), plus a single equalization coefficient `feq_coeff = 0.5*deq*( -Y2eq(left) + Y2eq(a) )`. Then, for each component `j`, it builds the three right-hand side pieces (`f0` non-interacting, `fL` linear if `coup_control` → `0.5*JS`, `feq = feq_coeff*S`) and writes the midpoint `DS(a,j) = S(a,j) + 0.5*dt*(f0R + fLR + feqR)` (and the imaginary part similarly). So `DS` is the half-step prediction.

The equalizer midpoint: if `eq_control` it computes `diffS2 = 0.5*deq*( -S2(a) + S2(next) )` and writes `DY(a,0) = Y(a,0) + 0.5*dt*diffS2*Y(a,0)` and `DY(a,1) = Y(a,1) - 0.5*dt*diffS2*Y(a,1)` — the signs follow from the `Y0`/`Y1` signed-norm structure. Finally it computes the midpoint norms `DS2` in a small sequential loop (not worth parallelizing).

**UPDATE PHASE:** it computes `J*DS` once into `JS` (the source is now the midpoint), then repeats the same per-spin structure but with `DS` as the state: the coefficients use the `DS2` norms, the equalization coefficient uses the `DY` signed norms of `left` and `a` (computed inline, not via `Y2eq_of_spin`), and the full-step update is `S(a,j) += dt*(f0R + fLR + feqR)`. This is the RK2 completion — it takes the slope evaluated at the midpoint and applies a full `dt` step from the **ORIGINAL** `S`. The equalizers get their full update with the `DY`-based `diffS2`, and finally `compute_hyperspin_norm()` refreshes `S2` for the next step / the `H` computation. Both phases parallelize over spins `a` with the `if(use_parallel)` gate; the ring topology means each spin only touches its own neighbours' norms, which are read before being written in this phase, keeping the region race-free (spin `a`'s equalizers are only written by spin `a`; `left`/`next` are only **READ**).

---

### 7.6) Physical and Diagnostic Functions

---

**`double max_eigenvalue_symm(double *data, int n)`**

Returns the **LARGEST** eigenvalue of a **SYMMETRIC** real n × n matrix stored row-major in `data` — a wrapper over GSL's symmetric eigensolver. It wraps the data in a `gsl_matrix_view` (no copy), allocates the eigen workspace, runs `gsl_eigen_symm`, scans the returned vector for the maximum, and frees the workspace. It needs GSL and a symmetric matrix; the callers always build symmetric matrices, so this is safe.

---

**`double Linearized_Max_Eigenvalue(int N, double *J_re, double *J_im, double *h, double g)`**

Returns the largest eigenvalue of the **LINEARIZED** (around the origin, zero-field) system matrix — the tool the code uses to measure whether the trivial state is unstable (the `Jmax`/`Lmax` analysis, and the `h_threshold` search). The full complex linear system has `2N` real degrees of freedom, so it builds a `2N × 2N` real symmetric matrix `Lmat` from a block structure:

- **top-left** N × N block: diagonal `0.5*(0.5*h[i] - g)`, off-diagonal `0.5*J_re`
- **bottom-right** N × N block: diagonal `0.5*(-0.5*h - g)`, off-diagonal `0.5*J_re`
- **cross block** coupling the im sector (rows >= N) to the re sector (cols < N): `-0.5*J_im`, mirrored symmetrically (that is the Hermitian structure)

It fills only the lower triangle (`j <= i`) and mirrors to the upper, then calls `max_eigenvalue_symm()` and frees. It builds the FULL `Lmat` fresh each call — fine for diagnostics, not a hot path.

---

**`double h_threshold(int N, double g, double *J_re, double *J_im)`**

Finds the pump threshold `hth` — the smallest uniform pump for which the linearized system just becomes unstable (`lmax` crosses zero) — and prints both `hth` and `lmax`. It is deliberately brute-force (see the `.c` guide's note): start `h_val = 0`, step `dh_val = 1e-5`, cap `hmax = 5`, then loop over the SAME `2N × 2N` `Lmat` structure as `Linearized_Max_Eigenvalue` but with `h_val` in the diagonal: build it, get `lmax` via `max_eigenvalue_symm()`, and keep stepping `h_val += dh_val` while `lmax <= 0`. Stop when `lmax` finally goes positive (or `h_val` reaches `hmax`), and return `hth = h_val - dh_val` — one step back, the last `h` with `lmax <= 0`. That is a brute-force linear sweep, not root-finding.

---

**`observables compute_observables(system_state *s, double g, double beta, double dt, double *L_old, int compute_H, int compute_full)`**

Computes the diagnostics with a two-level gate (the v2.9 restructure). When `compute_H = 0` and `compute_full = 0` the `.c` skips the call entirely (idle steps). When `compute_H = 1` and `compute_full = 0`, it computes exactly `H`. When `compute_full = 1` it also computes `L` (via `L0` and `Lc`), `dL`, `m`, and `DS2`, and returns an `observables` struct. The `.c` calls it every step with the appropriate flags: `compute_H = 1` on save-steps multiples, E-target runs, stability-check-aligned steps and the final step (the stop conditions need `H`); `compute_full = 1` only on `save_steps` multiples (the v2.4 optimization).

First, in parallel, it fills the shared scratch `inv_sqrtS2[i] = 1/sqrt(S2(i))`, computed once so both the `H` pass and the magnetization pass reuse it. If `compute_full`, a merged spin pass computes `L0` and the amplitude-variance pieces in **ONE** loop (this is the "merged loops reuse intermediates" optimization). For each `a` it accumulates `Lksqsum = sum_j (Re^2 - Im^2)` over the components, and:

- `L0 += -0.5*( 0.5*h[a]*Lksqsum - g*s2 - 0.5*beta*s2^2 )` (summed with a minus, since the code accumulates `L0_acc` with `-=`)

while also accumulating `mean_S2` and `mean2_S2`. Then `o.L0 = L0_acc`, and the amplitude variance is `DS2 = |mean2_S2 - mean_S2^2|/N`.

A merged EDGE pass (always run) computes `H` through the normalized fields `sR = S*inv_sqrtS2`:

- `H_acc -= 2*( Jr*(sRp*sRq + sIp*sIq) - Ji*(sRp*sIq - sIp*sRq) )` summed over the edge's components

and whenever `compute_full` it ALSO accumulates:

- `Lc_acc -= ( Jr*(srp*srq + sip*siq) + Ji*(sip*srq - srp*siq) )`

So `H` (whenever `compute_H=1`, the v2.9 gate) and `Lc` (only when `compute_full`) are computed in **THE SAME** edge loop — the merge optimization. `H` and `Lc` are stored; `Lc` stays **0** unless `compute_full`.

If `compute_full`: `o.L = o.L0 + o.Lc`, `o.dL = (o.L - *L_old)/dt`, and `*L_old = o.L` — the caller passes `&L_old` so `L` keeps its previous value for the derivative, which is why the `.c` declares a persistent `L_old`. Then the magnetization pass: it is parallelized over spins `a` (**NOT** over the D components — an OpenMP 4.5 VLA array reduction `compR[:D]`, `compI[:D]` accumulates `S*inv_sqrtS2` per component), and finally `m = sqrt( sum_k (compR^2 + compI^2) ) / N`.

At the end it returns the filled `observables` struct. `H` and `Lc` are always set; the rest are only meaningfully set when `compute_full = 1` (they read as **0** otherwise — the `.c` only uses them on save steps anyway).

---

### 7.7) Output Management (Buffered I/O)

---

**`io_context init_output_files(int N, int D, int save_dyn, int buffer_size, const char *base_dir)`**

Opens all output files for the chosen `save_dyn` mode and returns the `io_context` bundle. `base_dir` is the results directory when `output_dir = 1` (the `.c` passes `result_dir`), else `NULL`/empty, which puts the files in the current directory. It opens the files with the correct headers and sets up the buffering. The file layout by `save_dyn` (match this to Part 1's description):

- **save\_dyn 0:** opens **ONLY** `Data_Final.dat` in **APPEND** mode (`"a"`) — because in this mode every run appends a final row — and writes the header `NS  T  S1 Re(A11) Im(A11) ...`
- **save\_dyn 1:** opens `Data_Fields.dat`, `Data_Equalizers.dat`, `Data_Hamilton.dat` and `Data_Lyapunov.dat` in **WRITE** mode (`"w"`) with their headers, plus `Data_Final.dat` (`"w"`)
- **save\_dyn 2:** opens `Data_Hamilton.dat`, `Data_Lyapunov.dat` and `Data_Final.dat`
- **save\_dyn 3:** opens `Data_H_Ly_m_DS2.dat` and `Data_Final.dat`

Each opened file gets a `malloc`'d buffer of `buffer_size` and is switched to full buffering via `setvbuf(_IOFBF)` — that is the *"buffer_size / auto-flush when full"* mechanism. On any `fopen` failure it returns the partially-filled `ctx` (with the failed handle `NULL`); the `.c` checks `ctx.f_final` (plus `save_dyn == 0`) to report failure.

---

**`void save_output_files(io_context *ctx, int save_dyn, system_state *s, double t, double L, double L0, double Lc, double dL, double H, double m, double DS2)`**

Writes one time-sample of the observables (and, for `save_dyn = 1`, the full field states) to the open files. The `.c` calls it on `save_steps` multiples with the values that `compute_observables` returned:

- **save\_dyn 1:** writes the FULL field state (`t`, spin_id, `S2` plus all Re/Im per spin) to `f_fields` and the equalizer state (`t`, then the Re(Y) pairs, then the Im(Y) pairs) to `f_equalizers`, plus the `H` row and the Lyapunov row
- **save\_dyn 1 or 2:** writes `H` to `f_hamilton` and `(t, L, dL, L0, Lc)` to `f_lyapunov`
- **save\_dyn 3:** writes `(t, H, L/N, m, DS2)` to `f_h_ly_m_ds2` — note `L` is divided by `N` here (per-spin Lyapunov), and the columns are exactly `t H Ly/N <m> <DS2>` as the header says

---

**`void flush_output_files(io_context *ctx, int save_dyn)`**

`fflush`es every handle for the active mode plus `f_final` always, forcing the buffered data to disk — that is the `flush_interval` mechanism.

---

**`void close_output_files(io_context *ctx, int save_dyn)`**

`fclose`s every open handle for the active mode and `f_final`. Only the handles that were actually opened for that `save_dyn` are touched (guarded so `NULL` is never `fclose`'d).

---

**`void save_final_config(io_context *ctx, system_state *s, double t, int NS)`**

Appends the final configuration to `Data_Final.dat`: a single row `NS  T  ` followed, for every spin, by `S2(a)` and the full field values (Re/Im per component), then flushes. This is the "final state" row in every mode (in `save_dyn = 0` it is the **ONLY** thing written).

---

### 7.8) Hash and Output-Directory Helpers

---

**`char *read_file_to_string(const char *path)`**

Reads an entire file into a `malloc`'d, null-terminated C string; the caller must free it. It returns `NULL` on failure (cannot open, or empty). `compute_hash` uses it to fold file contents into the hash.

---

**`unsigned long long compute_hash(const char *ini_path, int seed, const char *matrix_path)`**

Computes a 64-bit FNV-1a hash of the ini file contents + the seed + the matrix file contents; that hash uniquely tags a run (it is how the `run_<hash>` directory is named). Standard FNV-1a: `out = offset_basis` (14695981039346656037ULL); for each byte of each ingredient, `out ^= byte`; `out *= prime` (1099511628211ULL). It feeds the ini file bytes, then the seed as a decimal string, then the matrix bytes, and returns the 64-bit hash. (The `.c` ALSO folds `pth` into the hash AFTER this returns — see Part 1, step 10.)

---

**`void format_hash_hex(unsigned long long hash, char hex[17])`**

Converts the 64-bit hash to a 16-character lowercase hexadecimal string in `hex`, which must be 17 bytes (16 chars + NUL). This is the printable run tag. It peels off nibbles from the least significant end, formats each as `'0'-'9'/'a'-'f'`, and places them so the string ends up big-endian (the most significant nibble first).

---

**`int copy_file_to_dir(const char *src, const char *dir, const char *dst_name)`**

Copies the file `src` to `dir/dst_name` (or just `dst_name` if `dir` is empty). Returns **0** on success, **1** on any failure. The `.c` uses it to copy `Complex_HSM.ini` and the matrix files into the `run_<hash>` directory. The implementation is a straightforward `fopen`/`fread`/`fwrite` in 8KB chunks, with failure checks on every file operation.

---

**`void write_metadata(const char *dir, int NS, int N, int D, int initial_type, double pth, double hm, double H, double t_final, const char *stop_reason)`**

Writes `metadata.txt` in the results directory documenting the run: the version string (Complex_HSM v2.9 — keep it in sync with the banner and the file headers when bumping versions), a timestamp, the key inputs (`NS`, `N`, `D`, `initial_type`, `pth`, `hm`), the final energy and time, the stop reason, and a list of the output files. It returns nothing, and silently does nothing if `dir` is empty/`NULL`, so it is safe to call even in exit-dir mode.

---

# Appendix

---

## A) Cross-Reference: `.c` Workflow Steps → Library Functions

If you read Part 1's numbered workflow, this tells you which library function each step actually calls:

| Step | Library Function(s) |
|------|---------------------|
| Step 1 | `parse_config()` |
| Step 3 | `print_header_and_ini_parameters()` |
| Step 5 | `allocate_system()` |
| Step 7 | `initialize_fields()`, `compute_hyperspin_norm()` |
| Step 8 | `initialize_J_Hermitian_Lattice` / `_Binary_Sparse` / `_Sparse_From_(Sparse_)File`; `fill_dense_from_sparse()`; `write_sparse`/`write_dense_matrix_file()` |
| Step 9 | `apply_coupling_rotation()`; `write_sparse_matrix_file()` (+ `write_dense`) |
| Step 10 | `compute_hash()`, `format_hash_hex()`, `copy_file_to_dir()` |
| Step 11 | `h_threshold()` |
| Step 12 | `initialize_pump()` (×2: h and phi) |
| Step 13 | `precompute_trig()` |
| Step 14 | `initialize_pump(&s, s.hmod, 0, 0)` (zero pump); `Linearized_Max_Eigenvalue()` (×2: for Jmax and Lmax) |
| Step 15 | `initialize_pump(&s, s.hmod, Dh, 0)` |
| Step 16 | `init_output_files()` |
| Step 17 | `print_memory_peak()` |
| Step 20.2 | `modify_pump()` |
| Step 20.3 | `compute_observables()`; `save_output_files()`; `flush_output_files()` |
| Step 20.4 | `RK2_step()` |
| Step 22 | `save_final_config()` |
| Step 23 | `write_metadata()` |
| Step 24 | `close_output_files()` |
| Step 26 | `free_system()` (+ the `fclose` of `f_final` in `save_dyn=0` mode) |

---

## B) Quick Array-Size Reference

Memorize these — they appear in every `malloc`/`calloc`:

| Size | Arrays | Notes |
|------|--------|-------|
| **N\*D** | `S_re`, `S_im`, `DS_re`, `DS_im`, `JS_re`, `JS_im` | Field arrays |
| **N** | `S2`, `DS2`, `h`, `hmod`, `phi`, `cos_phi`, `sin_phi`, `inv_sqrtS2` | Per-spin |
| **N\*2** | `Y_re`, `Y_im`, `DY_re`, `DY_im` | Equalizers — ALWAYS 2 per spin |
| **N\*N** | `J_re`, `J_im` | Dense J — **ONLY** if `compute_eigenvalues` |
| **N+1** | `csr_row_ptr` | CSR row pointer |
| **nnz** | `csr_col_idx`, `csr_Jr`, `csr_Ji` | `nnz = 2*num_edges` |
| **num\_edges** | `edge_a`, `edge_b`, `edge_Jr`, `edge_Ji` | `num_edges = nnz/2` |
| **2\*N** | linearized / threshold matrices | 2N real DOF |

---

*End of guide.*
