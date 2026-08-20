# Header layout

`HPRLP.h` is the stable public umbrella. The remaining canonical headers mirror
the implementation ownership under `src/`:

```text
include/
├── HPRLP.h
├── api/
├── batch/
├── io/
├── gpu/
│   ├── memory/
│   └── preprocessing/
│       ├── policies/
│       └── operators/
│           ├── common/
│           ├── dictionary/
│           ├── unit/
│           └── structured/
├── presolve/
├── solver/
│   ├── backends/
│   ├── graph/
│   ├── internal/
│   └── iteration/
├── support/
├── cuda_kernels/
```

Use canonical paths for implementation code, for example:

```cpp
#include "gpu/preprocessing/preprocess.h"
#include "solver/iteration/main_iterate.h"
#include "cuda_kernels/backends/unit/unit_kernels.cuh"
```

Historical flat header names are intentionally unsupported. Use the canonical paths
shown above; the root-level `HPRLP.h` public entry point remains unchanged.
