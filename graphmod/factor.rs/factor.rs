
trait ValueMapping {
	fn val_to_str(&self, val: uint) -> ~str;
	fn str_to_val(&self, &str) -> uint;
}

type ValMap<'self> = 'self |uint| -> ~str;

struct Type {
	name: ~str,
	nvalues: uint
}

struct Var {
	name: ~str,
	typ: Type
}

struct Value {
	typ: Type,
	val: uint
}

struct Instantiation {
	vars: ~[Var],
	values: ~[Value]
}

struct Factor {
	nvars: int
}

#[test]
fn trivial_test() {
	let x = 2;
	println!("This is a test {}", x)
}
