all: factor

factor: factor.rs
	rustc -o factor factor.rs

test: factor.rs
	rustc -o test --test factor.rs

clean: 
	rm -f factor
	rm -f test
