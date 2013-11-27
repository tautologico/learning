
trait ValueMapping {
	fn val_to_str(&self, val: uint) -> ~str;
	fn str_to_val(&self, &str) -> uint;
}

type ValMap<'self> = 'self |uint| -> ~str;

#[deriving(Eq)]
struct Type {
	name: ~str,
	nvalues: uint
}

#[deriving(Eq)]
struct Var {
	name: ~str,
	typ: Type
}

struct Value {
	typ: Type,
	val: uint
}

type Inst = ~[(Var, Value)];

struct Instantiation {
	vars: ~[Var],
	values: ~[Value]
}

struct Factor {
	normalized: bool,
	vars: ~[Var],
	values: ~[Value]
}

// --- operations ---------------------------------------------------

/// Returns variables in factor f that are not in vars
fn sum_factor_vars(f: &Factor, vars: &[Var]) -> ~[Var] {
	f.vars.iter().filter(|x| !vars.contains(*x)).collect()
}

// fn sum_out_vars(f: &Factor, vars: &[Var]) -> Factor {
// 	let res_vars = sum_factor_vars(f, vars);
// 	// TODO
// }

// fn multiply_factors(f1: &Factor, f2: &Factor) -> Factor {
// 	// TODO 
// }

#[test]
fn trivial_test() {
	let x = 2;
	println!("This is a test {}", x)
}
