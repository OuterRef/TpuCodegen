model:
	riscv64-unknown-linux-gnu-gcc -o model.elf model.c -I./csi-nn2/include ./csi-nn2/lib/libshl_c906.a -lm -static

clean:
	rm -rf *.elf
