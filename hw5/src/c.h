// Only the LAPACK / CBLAS bits we actually use. The full <Accelerate/Accelerate.h>
// umbrella currently crashes Zig 0.16's aro-based translate-c, so we go
// directly to vecLib (a subframework of Accelerate).
#define ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#include <vecLib/lapack.h>
