/**
 * @file version.h
 * @brief Version information for HPRLP library
 * 
 * This file contains version numbers and build information for the HPRLP library.
 * Version follows Semantic Versioning (https://semver.org/)
 */

#ifndef HPRLP_VERSION_H
#define HPRLP_VERSION_H

// Version numbers
#define HPRLP_VERSION_MAJOR 0
#define HPRLP_VERSION_MINOR 1
#define HPRLP_VERSION_PATCH 0

// Version string
#define HPRLP_VERSION_STRING "0.1.1"

// Full version number as integer (MAJOR * 10000 + MINOR * 100 + PATCH)
#define HPRLP_VERSION_NUMBER 101

// Build information
#define HPRLP_BUILD_DATE __DATE__
#define HPRLP_BUILD_TIME __TIME__

// API compatibility check macro
#define HPRLP_VERSION_CHECK(major, minor, patch) \
    ((HPRLP_VERSION_MAJOR > (major)) || \
     (HPRLP_VERSION_MAJOR == (major) && HPRLP_VERSION_MINOR > (minor)) || \
     (HPRLP_VERSION_MAJOR == (major) && HPRLP_VERSION_MINOR == (minor) && HPRLP_VERSION_PATCH >= (patch)))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the version string
 * @return Version string in format "MAJOR.MINOR.PATCH"
 */
inline const char* hprlp_get_version() {
    return HPRLP_VERSION_STRING;
}

/**
 * Get the major version number
 * @return Major version number
 */
inline int hprlp_get_version_major() {
    return HPRLP_VERSION_MAJOR;
}

/**
 * Get the minor version number
 * @return Minor version number
 */
inline int hprlp_get_version_minor() {
    return HPRLP_VERSION_MINOR;
}

/**
 * Get the patch version number
 * @return Patch version number
 */
inline int hprlp_get_version_patch() {
    return HPRLP_VERSION_PATCH;
}

/**
 * Get the build date
 * @return Build date string
 */
inline const char* hprlp_get_build_date() {
    return HPRLP_BUILD_DATE;
}

/**
 * Get the build time
 * @return Build time string
 */
inline const char* hprlp_get_build_time() {
    return HPRLP_BUILD_TIME;
}

#ifdef __cplusplus
}
#endif

#endif /* HPRLP_VERSION_H */
