/*
 * Copyright 2002-2011 Guillaume Cottenceau and contributors.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <iterator>

#define PNG_DEBUG 3
#include <png.h>


inline void abort_(const char * s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}



class PngImage;


template<bool flag, typename T, typename U>
    struct Select { typedef T Result; };

template<typename T, typename U>
    struct Select<false, T, U> { typedef U Result; };






template <bool isConst>
class PngImageIterator
{
public:
    typedef double value_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef double* pointer;
    typedef double& reference;


    PngImageIterator (const PngImage& image, bool isEnd = false);
    PngImageIterator (const PngImageIterator& other);
    PngImageIterator& operator= (const PngImageIterator& other);
    bool operator== (const PngImageIterator<isConst>& other) const;
    bool operator!= (const PngImageIterator<isConst>& other) const { return !((*this) == other); }
    PngImageIterator& operator++ ();
    PngImageIterator operator++ (int);
    value_type operator* ();


private:
    const PngImage& m_image;

    png_byte* m_pRow;
    png_byte* m_pPixel;
    int m_bytesPerPixel;
    int m_x;
    int m_y;
    int m_xMax;
    int m_yMax;
    
};



class PngImage
{


    int x, y;

    int m_width, m_height, rowbytes;
    png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes;
public:
    png_bytep * row_pointers;

    bool operator== (const PngImage& other) const
    {
        return row_pointers == other.row_pointers;
    }


    typedef PngImageIterator<false> iterator;

    iterator begin () { return iterator (*this, false); }
    iterator end   () { return iterator (*this, true); }



    void read_file(char* file_name)
    {
        unsigned char header[8];    // 8 is the maximum size that can be checked

        /* open file and test for it being a png */
        FILE *fp = fopen(file_name, "rb");
        if (!fp)
            abort_("[read_png_file] File %s could not be opened for reading", file_name);
        fread(header, 1, 8, fp);
        if (png_sig_cmp(header, 0, 8))
            abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


        /* initialize stuff */
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr)
            abort_("[read_png_file] png_create_read_struct failed");

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
            abort_("[read_png_file] png_create_info_struct failed");

        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during init_io");

        png_init_io(png_ptr, fp);
        png_set_sig_bytes(png_ptr, 8);

        png_read_info(png_ptr, info_ptr);

        m_width = png_get_image_width(png_ptr, info_ptr);
        m_height = png_get_image_height(png_ptr, info_ptr);
        color_type = png_get_color_type(png_ptr, info_ptr);
        bit_depth = png_get_bit_depth(png_ptr, info_ptr);

        number_of_passes = png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);


        /* read file */
        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during read_image");

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * m_height);

        if (bit_depth == 16)
            rowbytes = m_width*8;
        else
            rowbytes = m_width*4;

        for (y=0; y<m_height; y++)
            row_pointers[y] = (png_byte*) malloc(rowbytes);

        png_read_image(png_ptr, row_pointers);

        fclose(fp);
    }


    void write_file(char* file_name)
    {
        /* create file */
        FILE *fp = fopen(file_name, "wb");
        if (!fp)
            abort_("[write_png_file] File %s could not be opened for writing", file_name);


        /* initialize stuff */
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr)
            abort_("[write_png_file] png_create_write_struct failed");

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
            abort_("[write_png_file] png_create_info_struct failed");

        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during init_io");

        png_init_io(png_ptr, fp);


        /* write header */
        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during writing header");

        png_set_IHDR(png_ptr, info_ptr, m_width, m_height,
                     8, 6, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);


        /* write bytes */
        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during writing bytes");

        png_write_image(png_ptr, row_pointers);


        /* end write */
        if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during end of write");

        png_write_end(png_ptr, NULL);

        /* cleanup heap allocation */
        for (y=0; y<m_height; y++)
            free(row_pointers[y]);
        free(row_pointers);

        fclose(fp);
    }

    void grayscale ()
    {
        int error_action = 1;
        float red_weight = 0.3;
        float green_weight = 0.7;
        if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA)
            png_set_rgb_to_gray(png_ptr, error_action, red_weight, green_weight);
    }

    int height () const { return m_height; }
    int width () const { return m_width; }
    int pixels () const { return width () * height (); }
    int bytesPerPixel () const { return 1; }


    void process_file(void)
    {
        /* Expand any grayscale, RGB, or palette images to RGBA */
        png_set_expand(png_ptr);

        /* Reduce any 16-bits-per-sample images to 8-bits-per-sample */
        png_set_strip_16(png_ptr);

        for (y=0; y<m_height; y++) {
            png_byte* row = row_pointers[y];
            for (x=0; x<m_width; x++) {
                png_byte* ptr = &(row[x*4]);
                printf("Pixel at position [ %d - %d ] has RGBA values: %d - %d - %d - %d\n",
                       x, y, ptr[0], ptr[1], ptr[2], ptr[3]);

                /* perform whatever modifications needed, for example to set red value to 0 and green value to the blue one:
                   ptr[0] = 0;
                   ptr[1] = ptr[2]; */
            }
        }
    }


    friend class PngImageIterator<false>;
};




template <bool isConst>
PngImageIterator<isConst>::PngImageIterator (const PngImage& image, bool isEnd)
: m_image (image)
, m_pRow (image.row_pointers[(isEnd ? 0 : image.height ())])
    , m_pPixel (isEnd ? (png_byte*)NULL : &m_pRow[0])
    , m_bytesPerPixel (image.bytesPerPixel ())
    , m_x (isEnd ? image.width (): 0)
    , m_y (isEnd ? image.height (): 0)
    , m_xMax (image.width ())
    , m_yMax (image.height ())
{
}

template <bool isConst>
PngImageIterator<isConst>::PngImageIterator (const PngImageIterator& other)
: m_image (other.m_image)
, m_pRow (other.m_pRow)
    , m_pPixel (other.m_pPixel)
    , m_bytesPerPixel (other.m_bytesPerPixel)
    , m_x (other.m_x)
    , m_y (other.m_y)
    , m_xMax (other.m_xMax)
    , m_yMax (other.m_yMax)
{
}

template <bool isConst>
PngImageIterator<isConst>& PngImageIterator<isConst>::operator= (const PngImageIterator& other)
{
    m_image = other.m_image;
    m_pRow = other.m_pRow;
    m_pPixel = other.m_pPixel;
    m_bytesPerPixel = other.m_bytesPerPixel;
    m_x = other.m_x;
    m_y = other.m_y;
    m_xMax = other.m_xMax;
    m_yMax = other.m_yMax;
}



    template <bool isConst>
    bool PngImageIterator<isConst>::operator== (const PngImageIterator& other) const
    {
	return m_image == other.m_image &&
	m_pRow == other.m_pRow &&
	m_pPixel == other.m_pPixel &&
	m_bytesPerPixel == other.m_bytesPerPixel;
    }


template <bool isConst>
PngImageIterator<isConst>& PngImageIterator<isConst>::operator++ ()
{
    if (m_x < m_xMax)
    {
	++m_x;
	m_pPixel = &(m_pRow[m_bytesPerPixel*m_x]);
	return *this;
    }

    ++m_y;
    m_pRow = m_image.row_pointers[m_y];
    m_x = 0;
    m_pPixel = &(m_pRow[m_bytesPerPixel*m_x]);
    return *this;
}

template <bool isConst>
PngImageIterator<isConst> PngImageIterator<isConst>::operator++ (int)
{
    PngImageIterator tmp (*this);
    ++(*this);
    return tmp;
}

template <bool isConst>
typename PngImageIterator<isConst>::value_type PngImageIterator<isConst>::operator* ()
{
    return ((double)(*m_pPixel) /255.0);
}





/* int main(int argc, char **argv) */
/* { */
/*     if (argc != 3) */
/*         abort_("Usage: program_name <file_in> <file_out>"); */

/*     read_png_file(argv[1]); */
/*     process_file(); */
/*     write_png_file(argv[2]); */

/*     return 0; */
/* } */
