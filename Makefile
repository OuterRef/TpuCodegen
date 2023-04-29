conv2d:
	riscv64-unknown-linux-gnu-gcc -o conv2d.elf model.c -I/workspace/csi-nn2/include /workspace/csi-nn2/install_nn2/lib/libshl_c906.a -lm -static

clean:
	rm -rf *.elf