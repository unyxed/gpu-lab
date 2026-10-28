// The one translation unit that holds stb's function bodies.
// Everything else just #includes the headers without these macros.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
