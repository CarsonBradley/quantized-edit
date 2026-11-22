#!/bin/bash
# build.sh - hand-written equivalent of Strata's engine/src/inkling2/CMakeLists.txt + metal/CMakeLists.txt, for
# macOS machines without cmake or Xcode. Builds the unified `strata-inkling` (ink_main.cpp + Metal backend plugin),
# the same target the shipped 0.1.65 binary is built from. Needs only the Command Line Tools (clang), git and
# python3: shaders are embedded as source and compiled at runtime, so the `metal` tool is not needed.
#
# Usage: engine/build.sh [--patch] STRATA_DIR OUT_DIR
#   STRATA_DIR  a checkout (or export) of Strata at commit 26350e4 (the repository root, which contains engine/)
#   OUT_DIR     where objects and the binary go; the binary is OUT_DIR/strata-inkling
#   --patch     first apply, in this order, kldeval-64row-fix.patch and then ablate.patch (from this folder) to
#               STRATA_DIR; a patch that is already applied is skipped. ablate.patch needs the fix applied first.
#               Without --patch the tree is built as it is.
set -euo pipefail
PATCH=0
if [ "${1:-}" = "--patch" ]; then PATCH=1; shift; fi
if [ $# -ne 2 ]; then sed -n '2,14p' "$0"; exit 2; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $PATCH = 1 ]; then
  if git -C "$1" apply --reverse --check "$HERE/ablate.patch" 2>/dev/null; then
    echo "kldeval-64row-fix.patch and ablate.patch: already applied"
  else
    if git -C "$1" apply --reverse --check "$HERE/kldeval-64row-fix.patch" 2>/dev/null; then
      echo "kldeval-64row-fix.patch: already applied"
    else
      git -C "$1" apply "$HERE/kldeval-64row-fix.patch"; echo "kldeval-64row-fix.patch: applied"
    fi
    git -C "$1" apply "$HERE/ablate.patch"; echo "ablate.patch: applied"
  fi
fi
# Release-equivalent: CMake Release adds -DNDEBUG; shipped binary has LC_BUILD_VERSION minos 15.0.
REL=(-DNDEBUG -mmacosx-version-min=15.0)
E="$1/engine"
O="$2"
mkdir -p "$O/obj" "$O/zstd" "$O/gen"
Z="$E/third_party/zstd-dec"
I2="$E/src/inkling2"
INC=(-I"$E/include" -I"$I2" -I"$E/third_party/stb" -I"$Z" -I"$Z/common" -I"$Z/decompress")
# strata_inkling_core has PUBLIC INKLING_ENABLE_METAL=1 (set from metal/CMakeLists.txt), so every TU sees it.
DEFS=(-DINKLING_ENABLE_METAL=1)

# 1. vendored zstd (decompress only)
for f in common/debug.c common/entropy_common.c common/error_private.c common/fse_decompress.c common/xxhash.c \
         common/zstd_common.c decompress/huf_decompress.c decompress/zstd_ddict.c decompress/zstd_decompress.c \
         decompress/zstd_decompress_block.c; do
  b=$(basename "$f" .c)
  clang -O3 "${REL[@]}" -DZSTD_DISABLE_ASM=1 -I"$Z" -I"$Z/common" -I"$Z/decompress" -c "$Z/$f" -o "$O/zstd/$b.o"
done

# 2. strata_inkling_core (C++20, -O3; no OpenMP with Apple clang, same as find_package(OpenMP QUIET) failing)
for src in "$E/src/artifact/strata_reader.cpp" "$E/src/quant/sk_codecs.cpp" "$E/src/quant/dequant.cpp" \
           "$E/src/quant/types.cpp" "$I2/inkling2_loop.cpp" "$I2/ink_cpu_backend.cpp"; do
  b=$(basename "$src" .cpp)
  clang++ -std=c++20 -O3 "${REL[@]}" "${DEFS[@]}" "${INC[@]}" -c "$src" -o "$O/obj/core_$b.o"
done

# 3. embedded shaders header (runtime MSL compile; no metal/metallib tools needed)
python3 -I "$I2/metal/embed_shaders.py" "$I2/metal" "$O/gen/ink_metal_embedded_shaders.h"

# 4. strata_inkling_metal (OBJCXX C++20, -O3 -fobjc-arc)
MINC=(-I"$I2/metal" -I"$O/gen" -I"$E/include" -I"$E/src/hy4flash/run" -I"$E/metal/src" -I"$E/third_party/stb" -I"$Z" -I"$Z/common")
MDEFS=(-DINK_METAL_EMBEDDED_SHADERS=1 -DINK_METAL_HAS_MPP_SRC=1 '-DSTRATA_GGML_COMMON_DEFAULT=""' '-DSTRATA_METAL_KERNELS_DEFAULT=""' "${DEFS[@]}")
clang++ -x objective-c++ -std=c++20 -O3 "${REL[@]}" -fobjc-arc "${MDEFS[@]}" "${MINC[@]}" -c "$I2/metal/ink_metal_backend.mm" -o "$O/obj/ink_metal_backend.o"
clang++ -x objective-c++ -std=c++20 -O3 "${REL[@]}" -fobjc-arc "${MDEFS[@]}" "${MINC[@]}" -c "$E/metal/src/mtl_rt.mm" -o "$O/obj/mtl_rt.o"

# 5. strata-inkling executable
clang++ -std=c++20 -O3 "${REL[@]}" "${DEFS[@]}" "${INC[@]}" -c "$I2/ink_main.cpp" -o "$O/obj/ink_main.o"
clang++ "${REL[@]}" -o "$O/strata-inkling" "$O/obj/"*.o "$O/zstd/"*.o \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders -lpthread
ls -l "$O/strata-inkling"
