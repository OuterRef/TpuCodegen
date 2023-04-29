#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    FILE *fp;
    unsigned char byte[120];
    if (!(fp = fopen("./arrays.param", "rb")))
    {
        fprintf(stderr, "Open file failed.\n");
        return -1;
    }
    fread(&byte, 8, 120, fp);
    for (int i = 0; i < 120; i++)
    {
        printf("0x%x ", byte[i]);
    }
    return 0;
}
