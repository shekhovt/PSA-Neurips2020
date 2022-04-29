
#ifndef NDARRAY_EXPORTS_H
#define NDARRAY_EXPORTS_H

#ifdef SHARED_EXPORTS_BUILT_AS_STATIC
#  define NDARRAY_EXPORTS
#  define NDARRAY_NO_EXPORT
#else
#  ifndef NDARRAY_EXPORTS
#    ifdef ndarray_EXPORTS
        /* We are building this library */
#      define NDARRAY_EXPORTS __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define NDARRAY_EXPORTS __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef NDARRAY_NO_EXPORT
#    define NDARRAY_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef NDARRAY_DEPRECATED
#  define NDARRAY_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef NDARRAY_DEPRECATED_EXPORT
#  define NDARRAY_DEPRECATED_EXPORT NDARRAY_EXPORTS NDARRAY_DEPRECATED
#endif

#ifndef NDARRAY_DEPRECATED_NO_EXPORT
#  define NDARRAY_DEPRECATED_NO_EXPORT NDARRAY_NO_EXPORT NDARRAY_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef NDARRAY_NO_DEPRECATED
#    define NDARRAY_NO_DEPRECATED
#  endif
#endif

#endif
