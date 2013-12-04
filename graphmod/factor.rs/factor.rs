
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
	typ: Type
}

/// A value for a discrete variable is just an index into the list of values
type Value = uint;

type Assignment = ~[(Var, Value)];

struct Factor<'r> {
	vars: ~[Var],
	table: &'r HashSymbTable,
	values: ~[f64]
}

impl<'r> Factor<'r> {
	// fn empty_table(vars: ~[Var], vals: ~[f64]) -> Factor {
	// 	Factor { vars: vars, table: HashSymbTable::new(), values: vals }
	// }

	fn new(vars: ~[Var], table: &'r HashSymbTable, vals: ~[f64]) -> Factor<'r> {
		Factor { vars: vars, table: table, values: vals }
	}

	#[inline]
	fn index_of_var(&self, v: Var) -> Option<uint> {
		self.vars.position_elem(&v)
	}

	pub fn marginalize_vars(&self, vars: &[Var]) -> Factor<'r> {
		let res_vars = self.sum_factor_vars(vars);
		let vixs = self.vars_indices(res_vars);     // indices of result vars (rel. to self.vars)
		let mut res_vals = zero_vector_f64(self.card_vars(res_vars));
		let mut assign = zero_vector_uint(self.vars.len());

		for i in range(0, self.values.len()) {
			let mut ix = 0;
			for j in range(0, assign.len()) {
				match vixs.position_elem(&j) {
					Some(0) => ix += assign[j],
					Some(vi) => ix += assign[j] * self.table.var_cardinality(self.vars[vixs[vi-1]]),
					None => ()
				}
			}
			res_vals[ix] += self.values[i];
			self.next_assignment(assign);
		}

		Factor::new(res_vars, self.table, res_vals)
	}

	// --- private methods

	fn index_to_assignment(&self, ix: uint) -> ~[Value] {
		let mut res = std::vec::with_capacity(self.vars.len());
		let mut c = ix;
		for i in range(0, self.vars.len()) {
			let card = self.table.var_cardinality(self.vars[i]);
			res.push(c % card);
			c = c / card;
		}
		res
	}

	fn next_assignment(&self, assign: &mut [uint]) {
		for i in range(0, assign.len()) {
			let v = self.vars[i];
			assign[i] = (assign[i] + 1) % self.table.var_cardinality(v);
			if assign[i] != 0 {
				break;
			}
		}
	}

	/// Returns the indices of variables in vars among the factor variables
	fn vars_indices(&self, vars: &[Var]) -> ~[uint] {
		vars.iter().map(|x| self.vars.position_elem(x).unwrap()).collect()
	}

	/// Returns variables in factor that are not in vars
	fn sum_factor_vars(&self, vars: &[Var]) -> ~[Var] {
		self.vars.iter().filter(|x| !vars.contains(*x)).map(|&x| x.clone()).collect()
	}

	/// Returns the cardinality of a cartesian product of variables
	fn card_vars(&self, vars: &[Var]) -> uint {
		vars.iter().map(|&v| self.table.var_cardinality(v)).fold(1, |c1, c2| c1 * c2)
	}
}

// --- utilities ----------------------------------------------------

#[inline]
fn zero_vector_f64(n: uint) -> ~[f64] {
	std::vec::from_elem(n, 0.0)
}

#[inline]
fn zero_vector_uint(n: uint) -> ~[uint] {
	std::vec::from_elem(n, 0u)
}

// fn multiply_factors(f1: &Factor, f2: &Factor) -> Factor {
// 	// TODO 
// }


#[cfg(test)]
mod tests {
	use super::Factor;
	use super::HashSymbTable;

	#[inline]
	fn get_symb_table_1() -> HashSymbTable {
		let mut table = HashSymbTable::new();
		let booltyp = table.new_type(~"bool", 2);
		let _ = table.new_var(~"A", booltyp);
		let _ = table.new_var(~"B", booltyp);
		let _ = table.new_var(~"C", booltyp);
		table
	}

	#[inline]
	fn get_test_factor_1<'r>(table: &'r HashSymbTable) -> Factor<'r> {
		Factor::new(~[0, 1, 2], table,
			        ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
	}

	#[inline]
	fn get_symb_table_2() -> HashSymbTable {
	    let mut table = HashSymbTable::new();
		let booltyp = table.new_type(~"bool", 2);
		let trityp = table.new_type(~"triple", 3);
		let _ = table.new_var(~"A", booltyp);
		let _ = table.new_var(~"B", trityp);
		let _ = table.new_var(~"C", booltyp);
		table	
	}

	#[inline]
	fn get_test_factor_2<'r>(table: &'r HashSymbTable) -> Factor<'r> {
		Factor::new(~[0, 1, 2], table,
			        ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
	}

	#[test]
	fn test_marginalize_vars() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		let f2 = f1.marginalize_vars([1]);
		assert_eq!(f2.vars.len(), 2);
		assert_eq!(f2.values.len(), 4);
		assert_eq!(f2.values, ~[0.5, 0.5, 0.5, 0.5]);

		let table2 = get_symb_table_2();
		let f3 = get_test_factor_2(&table2);
		let f4 = f3.marginalize_vars([1]);
		assert_eq!(f4.vars.len(), 2);
		assert_eq!(f4.values.len(), 4);
		assert_eq!(f4.values, ~[0.75, 0.75, 0.75, 0.75]);
	}

	#[test]
	fn test_index_to_assignment() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		assert_eq!(f1.index_to_assignment(0), ~[0, 0, 0]);
		assert_eq!(f1.index_to_assignment(1), ~[1, 0, 0]);
		assert_eq!(f1.index_to_assignment(2), ~[0, 1, 0]);
		assert_eq!(f1.index_to_assignment(3), ~[1, 1, 0]);
		assert_eq!(f1.index_to_assignment(4), ~[0, 0, 1]);
		assert_eq!(f1.index_to_assignment(7), ~[1, 1, 1]);

		let table2 = get_symb_table_2();
		let f2 = get_test_factor_2(&table2);
		assert_eq!(f2.index_to_assignment(0),  ~[0, 0, 0]);
		assert_eq!(f2.index_to_assignment(1),  ~[1, 0, 0]);
		assert_eq!(f2.index_to_assignment(2),  ~[0, 1, 0]);
		assert_eq!(f2.index_to_assignment(3),  ~[1, 1, 0]);
		assert_eq!(f2.index_to_assignment(4),  ~[0, 2, 0]);
		assert_eq!(f2.index_to_assignment(5),  ~[1, 2, 0]);
		assert_eq!(f2.index_to_assignment(6),  ~[0, 0, 1]);
		assert_eq!(f2.index_to_assignment(11), ~[1, 2, 1]);
	}

	#[test]
	fn test_next_assignment() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		let mut assign = [0u, 0u, 0u];
		f1.next_assignment(assign);
		assert_eq!(assign, [1u, 0u, 0u]);
		f1.next_assignment(assign);
		assert_eq!(assign, [0u, 1u, 0u]);
		f1.next_assignment(assign);
		assert_eq!(assign, [1u, 1u, 0u]);
		f1.next_assignment(assign);
		assert_eq!(assign, [0u, 0u, 1u]);
	}

	#[test]
	fn test_vars_indices() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		let vs1 = ~[1];
		let vs2 = ~[0, 2];
		let vs3 = ~[0, 1, 2];

		assert_eq!(f1.vars_indices(vs1), ~[1]);
		assert_eq!(f1.vars_indices(vs2), ~[0, 2]);
		assert_eq!(f1.vars_indices(vs3), ~[0, 1, 2]);
	}

	#[test]
	fn test_card_vars() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		let vs1 = ~[1];
		let vs2 = ~[0, 2];
		let vs3 = ~[0, 1, 2];

		assert_eq!(f1.card_vars(vs1), 2);
		assert_eq!(f1.card_vars(vs2), 4);
		assert_eq!(f1.card_vars(vs3), 8);
	}

	#[test]
	fn test_sum_factor_vars() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
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
