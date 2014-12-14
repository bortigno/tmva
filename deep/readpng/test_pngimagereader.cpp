#include "readpng.h"





int main(int argc, char **argv)
{
    if (argc != 3)
        abort_("Usage: program_name <file_in> <file_out>");

    PngImage image;

    image.read_file(argv[1]);
    image.process_file();
    image.write_file(argv[2]);

    return 0;
}



