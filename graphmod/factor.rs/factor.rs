
#[feature(globs)];

use std::hashmap::HashMap;

trait ValueMapping {
	fn val_to_str(&self, val: uint) -> ~str;
	fn str_to_val(&self, &str) -> uint;
}

/// A discrete variable is just an ID for the variable specification in an index
type Var = uint;

/// Symbol table of variable and type specifications
trait SymbTable<'r> {
	fn var_spec(&'r self, v: Var) -> &'r VarSpec;
	fn type_spec(&'r self, t: Type) -> &'r TypeSpec;
	fn var_cardinality(&self, v: Var) -> uint;
}

struct HashSymbTable {
	var_index: HashMap<Var, VarSpec>,
	typ_index: HashMap<Type, TypeSpec>
}

impl HashSymbTable {
	fn new() -> HashSymbTable {
		HashSymbTable { var_index: HashMap::new(), typ_index: HashMap::new() }
	}

	fn new_type(&mut self, name: ~str, card: uint) -> Type {
		let k = self.typ_index.len() as Type;
		let typ = TypeSpec { name: name, cardinality: card };

		if !self.typ_index.insert(k, typ) {
			fail!("Error adding type to HashSymbTable: type already exists");
		}
		k
	}

	fn new_var(&mut self, name: ~str, t: Type) -> Var {
		let k = self.var_index.len() as Var;
		let vs = VarSpec { name: name, typ: t };

		if !self.var_index.insert(k, vs) {
			fail!("Error adding var to HashSymbTable: var already exists");
		}
		k
	}

	fn from_vars(vars: &[(Var, VarSpec)]) -> HashSymbTable {
		let mut ix = HashMap::with_capacity(vars.len());
		for &(ref v, ref vs) in vars.iter() {
			if !ix.insert(*v, vs.clone()) {
				fail!("Error inserting variable spec in HashSymbTable: var already existed");
			}
		}
		HashSymbTable{ var_index: ix, typ_index: HashMap::new() }
	}
}

impl<'r> SymbTable<'r> for HashSymbTable {
	fn var_spec(&'r self, v: Var) -> &'r VarSpec {
		self.var_index.get(&v)
	}

	fn type_spec(&'r self, t: Type) -> &'r TypeSpec {
		self.typ_index.get(&t)
	}

	fn var_cardinality(&self, v: Var) -> uint {
		let typ = self.var_index.get(&v).typ;
		self.typ_index.get(&typ).cardinality
	}
}

type Type = uint;

/// A discrete (categorical) type for variables
#[deriving(Eq, Clone)]
struct TypeSpec {
	name: ~str,
	cardinality: uint
}

#[deriving(Eq, Clone)]
struct VarSpec {
	name: ~str,
	typ: Type    // TODO: shared reference to Type?
}

/// A value for a discrete variable is just an index into the list of values
type Value = uint;

type Assignment = ~[(Var, Value)];

struct Factor {
	vars: ~[Var],
	table: HashSymbTable,
	values: ~[f64]
}

impl Factor {
	fn empty_table(vars: ~[Var], vals: ~[f64]) -> Factor {
		Factor { vars: vars, table: HashSymbTable::new(), values: vals }
	}

	fn new(vars: ~[Var], table: HashSymbTable, vals: ~[f64]) -> Factor {
		Factor { vars: vars, table: table, values: vals }
	}

	#[inline]
	fn index_of_var(&self, v: Var) -> Option<uint> {
		self.vars.position_elem(&v)
	}

	pub fn marginalize_vars(&self, vars: &[Var]) -> Factor {
		let res_vars = self.sum_factor_vars(vars);

		//for v in vars.iter() {

		//}

		Factor::new(res_vars, HashSymbTable::new(), ~[0.0, 1.0])
	}

	// --- private methods

	/// Returns variables in factor that are not in vars
	fn sum_factor_vars(&self, vars: &[Var]) -> ~[Var] {
		self.vars.iter().filter(|x| !vars.contains(*x)).map(|&x| x.clone()).collect()
	}

	/// Returns the cardinality of a cartesian product of variables
	fn card_vars(&self, vars: &[Var]) -> uint {
		vars.iter().map(|&v| self.table.var_cardinality(v)).fold(1, |c1, c2| c1 * c2)
	}
}

// --- operations ---------------------------------------------------


// fn multiply_factors(f1: &Factor, f2: &Factor) -> Factor {
// 	// TODO 
// }


#[cfg(test)]
mod tests {
	use super::Factor;
	use super::{VarSpec};
	use super::HashSymbTable;

	#[inline]
	fn get_test_factor_1() -> Factor {
		let mut f = Factor::empty_table(~[0, 1, 2], 
			                            ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]);
		let booltyp = f.table.new_type(~"bool", 2);
		let v1 = f.table.new_var(~"A", booltyp);
		let v2 = f.table.new_var(~"B", booltyp);
		let v3 = f.table.new_var(~"C", booltyp);
		f
	}

	#[test]
	fn test_card_vars() {
		let f1 = get_test_factor_1();
		let vs1 = ~[1];
		let vs2 = ~[0, 2];
		let vs3 = ~[0, 1, 2];

		assert_eq!(f1.card_vars(vs1), 2);
		assert_eq!(f1.card_vars(vs2), 4);
		assert_eq!(f1.card_vars(vs3), 8);
	}

	#[test]
	fn test_sum_factor_vars() {
		let f1 = get_test_factor_1();
		let vs1 = ~[1];
		let vs2 = ~[0];
		let vs3 = ~[0, 2];
		let vs4 = ~[0, 1, 2];

		assert_eq!(f1.sum_factor_vars(vs1), ~[0, 2]);
		assert_eq!(f1.sum_factor_vars(vs2), ~[1, 2]);
		assert_eq!(f1.sum_factor_vars(vs3), ~[1]);
		assert_eq!(f1.sum_factor_vars(vs4), ~[]);
	}
}
