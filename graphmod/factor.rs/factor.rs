
#[feature(globs)];

use std::hashmap::HashMap;

trait ValueMapping {
	fn val_to_str(&self, val: uint) -> ~str;
	fn str_to_val(&self, &str) -> uint;
}

/// A discrete variable is just an ID for the variable specification in a SymbolTable
type Var = uint;

/// A discrete type is just an ID for the type specification in a SymbolTable
type Type = uint;

/// A value for a discrete variable is just an index into the list of values
type Value = uint;

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

/// Symbol table of variable and type specifications
trait SymbTable<'r> {
	fn var_spec(&'r self, v: Var) -> &'r VarSpec;
	fn type_spec(&'r self, t: Type) -> &'r TypeSpec;
	fn var_cardinality(&self, v: Var) -> uint;
	fn vars_cardinality(&self, vars: &[Var]) -> uint;
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

	fn vars_cardinality(&self, vars: &[Var]) -> uint {
		vars.iter().map(|&v| self.var_cardinality(v)).fold(1, |c1, c2| c1 * c2)
	}
}

/// An assignment related to a set of variables 
// TODO: could be "Assignments", a collection of possible assignments
// for the variables vars; define an iterator over it (next())
struct Assignment<'r> {
	vars: &'r [Var],
	table: &'r HashSymbTable,
	values: ~[Value]
}

impl<'r> Assignment<'r> {
	fn zero(vars: &'r [Var], table: &'r HashSymbTable) -> Assignment<'r> {
		let res_vals = zero_vector_uint(vars.len());
		Assignment { vars: vars, table: table, values: res_vals }
	}

	fn from_index(vars: &'r [Var], table: &'r HashSymbTable, ix: uint) -> Assignment<'r> {
		let mut res = std::vec::with_capacity(vars.len());
		let mut c = ix;
		for i in range(0, vars.len()) {
			let card = table.var_cardinality(vars[i]);
			res.push(c % card);
			c = c / card;
		}
		
		Assignment { vars: vars, table: table, values: res }
	}

	/// project assignment to a smaller set of variables
	/// to_vars, and return a value index for the resulting assignment
	fn project_and_get_index(&self, to_vars: &[Var]) -> uint {
		let (ix, _) = 
		    self.values.iter()
			    .zip(self.vars.iter())
			    .filter(|&(_, var)| to_vars.contains(var))
			    .map(|(val, var)| (val, self.table.var_cardinality(*var)))			        
			    .fold((0, 1), |(sum, cs), (val, c2)| (sum + val * cs, c2 * cs));
		ix
	}

	fn set_values(&mut self, vals: &[Value]) {
		assert_eq!(self.values.len(), vals.len());
		for i in range(0, vals.len()) {
			self.values[i] = vals[i];
		}
	}

	fn to_index(&self) -> uint {
		self.project_and_get_index(self.vars)
	}

	fn next(&mut self) {
	    for i in range(0, self.values.len()) {
			let v = self.vars[i];
			self.values[i] = (self.values[i] + 1) % self.table.var_cardinality(v);
			if self.values[i] != 0 {
				break;
			}
		}
	}

	/// Sweep all assignments for variables var, executing f on each
	fn sweep(vars: &'r [Var], table: &'r HashSymbTable, f: |uint, &Assignment|) {
		let mut assign = Assignment::zero(vars, table);
		let max_index = table.vars_cardinality(vars);
		for ix in range(0, max_index) {
			f(ix, &assign);
			assign.next();
		}
	}
}

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
		let mut res_vals = zero_vector_f64(self.table.vars_cardinality(res_vars));

		Assignment::sweep(self.vars, self.table, 
			|i, assign| {
				let ix = assign.project_and_get_index(res_vars);
				res_vals[ix] += self.values[i];
			}
		);

		Factor::new(res_vars, self.table, res_vals)
	}

	/// Create a new factor by multiplying self with the other factor 
	/// (both must use the same symbol table)
	pub fn multiply(&self, other: &'r Factor) -> Factor<'r> {
		let res_vars = self.union_vars(other);
		let mut res_vals = std::vec::from_elem(self.table.vars_cardinality(res_vars), 1.0);

	    Assignment::sweep(res_vars, self.table,
	    	|i, assign| {
	    		let si = assign.project_and_get_index(self.vars);
	    		let oi = assign.project_and_get_index(other.vars);
	    		res_vals[i] *= self.values[si] * other.values[oi];
	    	}
	    );

		Factor::new(res_vars, self.table, res_vals)
	}

	pub fn multiply_many(&self, others: &'r [&'r Factor]) -> Factor<'r> {
		let res_vars = self.union_vars_many(others);
		let mut res_vals = std::vec::from_elem(self.table.vars_cardinality(res_vars), 1.0);

		Assignment::sweep(res_vars, self.table,
	    	|i, assign| {
	    		let si = assign.project_and_get_index(self.vars);
			    res_vals[i] *= self.values[si];
			    for f in others.iter() {
				    let oi = assign.project_and_get_index(f.vars);
				    res_vals[i] *= f.values[oi];
			    }
	    	}
	    );

		Factor::new(res_vars, self.table, res_vals)
	}

	// --- private methods

	/// Returns the indices of variables in vars among the factor variables
	fn vars_indices(&self, vars: &[Var]) -> ~[uint] {  // TODO: remove?
		vars.iter().map(|x| self.vars.position_elem(x).unwrap()).collect()
	}

	/// Returns variables in factor that are not in vars
	fn sum_factor_vars(&self, vars: &[Var]) -> ~[Var] {
		self.vars.iter().filter(|x| !vars.contains(*x)).map(|&x| x.clone()).collect()
	}

	/// Returns the union of the variables in self and other
	#[inline]
	fn union_vars(&self, other: &'r Factor) -> ~[Var] {
		var_union(self.vars, other.vars)
	}

	fn union_vars_many(&self, others: &'r [&'r Factor]) -> ~[Var] {
		let mut res_vars = self.vars.clone();
		for f in others.iter() {
			// PERF: many intermediate vectors generated
			res_vars = var_union(res_vars, f.vars); 
		}
		res_vars
	}
}

// --- utilities ----------------------------------------------------

fn var_union(v1: &[Var], v2: &[Var]) -> ~[Var] {
	v1.iter()
		.chain(
		    v2.iter().filter(|&v| !v1.contains(v)))
		.map(|&v| v.clone())
		.collect()
}

#[inline]
fn zero_vector_f64(n: uint) -> ~[f64] {
	std::vec::from_elem(n, 0.0)
}

#[inline]
fn zero_vector_uint(n: uint) -> ~[uint] {
	std::vec::from_elem(n, 0u)
}


#[cfg(test)]
mod tests {
	use super::Factor;
	use super::Assignment;
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
			        ~[0.25, 0.25, 0.25, 0.25, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.50])
	}

	#[inline]
	fn get_symb_table_3() -> HashSymbTable {
	    let mut table = HashSymbTable::new();
		let booltyp = table.new_type(~"bool", 2);
		let _ = table.new_var(~"A", booltyp);
		let _ = table.new_var(~"B", booltyp);
		let _ = table.new_var(~"C", booltyp);
		let _ = table.new_var(~"D", booltyp);
		table	
	}

	#[inline]
	fn get_test_factor_3<'r>(table: &'r HashSymbTable) -> Factor<'r> {
		Factor::new(~[0, 1, 2, 3], table,
			        ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
			          0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
	}

	#[test]
	fn test_marginalize_vars() {
		let table = get_symb_table_1();
		let f1 = get_test_factor_1(&table);
		let f1_1 = f1.marginalize_vars([1]);
		assert_eq!(f1_1.vars.len(), 2);
		assert_eq!(f1_1.values.len(), 4);
		assert_eq!(f1_1.values, ~[0.5, 0.5, 0.5, 0.5]);
		let f1_2 = f1.marginalize_vars([0, 1]);
		assert_eq!(f1_2.vars.len(), 1);
		assert_eq!(f1_2.values.len(), 2);
		assert_eq!(f1_2.values, ~[1.0, 1.0]);

		let table2 = get_symb_table_2();
		let f2 = get_test_factor_2(&table2);
		let f2_1 = f2.marginalize_vars([1]);
		assert_eq!(f2_1.vars.len(), 2);
		assert_eq!(f2_1.values.len(), 4);
		assert_eq!(f2_1.values, ~[1.0, 0.75, 0.75, 1.0]);
		let f2_2 = f2.marginalize_vars([0]);
		assert_eq!(f2_2.vars.len(), 2);
		assert_eq!(f2_2.values.len(), 6);
		assert_eq!(f2_2.values, ~[0.5, 0.5, 0.75, 0.5, 0.5, 0.75]);

		let table3 = get_symb_table_3();
		let f3 = get_test_factor_3(&table3);
		let f3_1 = f3.marginalize_vars([1]);
		assert_eq!(f3_1.vars.len(), 3);
		assert_eq!(f3_1.values.len(), 8);
		assert_eq!(f3_1.values, ~[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
	}

	#[test]
	fn test_multiply() {
		let table = get_symb_table_3();
		let f1_1 = Factor::new(~[0, 1], &table,
			                 ~[0.5, 0.5, 0.5, 0.5]);
		let f1_2 = Factor::new(~[1, 2], &table,
			                 ~[0.5, 0.5, 0.5, 0.5]);
		let fm1 = f1_1.multiply(&f1_2);
		assert_eq!(fm1.vars.len(), 3);
		assert_eq!(fm1.values.len(), 8);
		assert_eq!(fm1.values, ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]);

		let f2_1 = Factor::new(~[0, 1, 2], &table,
			                 ~[0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.0, 1.0]);
		let f2_2 = Factor::new(~[2, 3], &table,
			                 ~[0.448, 0.192, 0.112, 0.248]);
		let fm2 = f2_1.multiply(&f2_2);
		assert_eq!(fm2.vars.len(), 4);
		assert_eq!(fm2.vars, ~[0, 1, 2, 3]);
		assert_eq!(fm2.values.len(), 16);
		assert_eq!(fm2.values, ~[0.4256, 0.05 * 0.448, 0.9 * 0.448, 0.1 * 0.448,
			                     0.8 * 0.192, 0.2 * 0.192, 0.0, 0.192,
			                     0.95 * 0.112, 0.05 * 0.112, 0.9 * 0.112, 0.1 * 0.112,
			                     0.8 * 0.248, 0.2 * 0.248, 0.0, 0.248]);
	}

	#[test]
	fn test_multiply_many() {
		let table = get_symb_table_3();
		let f1_1 = Factor::new(~[0, 1], &table,
			                   ~[0.5, 0.5, 0.5, 0.5]);
		let f1_2 = Factor::new(~[1, 2], &table,
			                   ~[0.4, 0.2, 0.2, 0.4]);
		let f1_3 = Factor::new(~[2, 3], &table,
			                   ~[0.1, 0.3, 0.5, 0.4]);
		let factors = [&f1_2, &f1_3];
		let fm1 = f1_1.multiply_many(factors);
		assert_eq!(fm1.vars.len(), 4);
		assert_eq!(fm1.values.len(), 16);
		assert_eq!(fm1.values, ~[0.5 * 0.4 * 0.1, 0.5 * 0.4 * 0.1, 0.5 * 0.2 * 0.1, 0.5 * 0.2 * 0.1,
								 0.5 * 0.2 * 0.3, 0.5 * 0.2 * 0.3, 0.5 * 0.4 * 0.3, 0.5 * 0.4 * 0.3,
								 0.5 * 0.4 * 0.5, 0.5 * 0.4 * 0.5, 0.5 * 0.2 * 0.5, 0.5 * 0.2 * 0.5,
								 0.5 * 0.2 * 0.4, 0.5 * 0.2 * 0.4, 0.5 * 0.4 * 0.4, 0.5 * 0.4 * 0.4]);
	}

	#[test]
	fn test_index_to_assignment() {
		let table = get_symb_table_1();
		let vars = [0, 1, 2];
		let a1 = Assignment::from_index(vars, &table, 0u);
		assert_eq!(a1.values, ~[0, 0, 0]);
		let a2 = Assignment::from_index(vars, &table, 1u);
		assert_eq!(a2.values, ~[1, 0, 0]);
		let a3 = Assignment::from_index(vars, &table, 2u);
		assert_eq!(a3.values, ~[0, 1, 0]);
		let a4 = Assignment::from_index(vars, &table, 3u);
		assert_eq!(a4.values, ~[1, 1, 0]);
		let a5 = Assignment::from_index(vars, &table, 4u);
		assert_eq!(a5.values, ~[0, 0, 1]);
		let a6 = Assignment::from_index(vars, &table, 7u);
		assert_eq!(a6.values, ~[1, 1, 1]);

		let table2 = get_symb_table_2();
		let a7 = Assignment::from_index(vars, &table2, 0u);
		assert_eq!(a7.values,  ~[0, 0, 0]);
		let a8 = Assignment::from_index(vars, &table2, 1u);
		assert_eq!(a8.values,  ~[1, 0, 0]);
		let a9 = Assignment::from_index(vars, &table2, 2u);
		assert_eq!(a9.values,  ~[0, 1, 0]);
		let a10 = Assignment::from_index(vars, &table2, 3u);
		assert_eq!(a10.values,  ~[1, 1, 0]);
		let a11 = Assignment::from_index(vars, &table2, 4u);
		assert_eq!(a11.values,  ~[0, 2, 0]);
		let a12 = Assignment::from_index(vars, &table2, 5u);
		assert_eq!(a12.values,  ~[1, 2, 0]);
		let a13 = Assignment::from_index(vars, &table2, 6u);
		assert_eq!(a13.values,  ~[0, 0, 1]);
		let a14 = Assignment::from_index(vars, &table2, 11u);
		assert_eq!(a14.values, ~[1, 2, 1]);
	}

	#[test]
	fn test_assignment_to_index() {
		let table = get_symb_table_1();
		let vars = [0, 1, 2];
		let mut assign = Assignment::zero(vars, &table);
		assert_eq!(assign.to_index(), 0u);
		assign.next();
		assert_eq!(assign.to_index(), 1u);
		assign.next();
		assert_eq!(assign.to_index(), 2u);
		assign.set_values([0, 0, 1]);
		assert_eq!(assign.to_index(), 4u);
		assign.set_values([0, 1, 1]);
		assert_eq!(assign.to_index(), 6u);
		assign.next();
		assert_eq!(assign.to_index(), 7u);

		let table2 = get_symb_table_2();
		let mut assign2 = Assignment::zero(vars, &table2);
		assert_eq!(assign2.to_index(), 0u);
		assign2.next();
		assert_eq!(assign2.to_index(), 1u);
		assign2.next();
		assert_eq!(assign2.to_index(), 2u);
		assign2.set_values([0, 2, 0]);
		assert_eq!(assign2.to_index(), 4u);
		assign2.set_values([0, 1, 1]);
		assert_eq!(assign2.to_index(), 8u);
		assign2.next();
		assert_eq!(assign2.to_index(), 9u);
		assign2.set_values([0, 2, 1]);
		assert_eq!(assign2.to_index(), 10u);
		assign2.next();
		assert_eq!(assign2.to_index(), 11u);
	}

	#[test]
	fn test_assignment_next() {
		let table = get_symb_table_1();
		let vars = ~[0, 1, 2];
		let mut assign = Assignment::zero(vars, &table);
		assign.next();
		assert_eq!(assign.values, ~[1u, 0u, 0u]);
		assign.next();
		assert_eq!(assign.values, ~[0u, 1u, 0u]);
		assign.next();
		assert_eq!(assign.values, ~[1u, 1u, 0u]);
		assign.next();
		assert_eq!(assign.values, ~[0u, 0u, 1u]);
		assign.next();
		assign.next();
		assign.next();
		assert_eq!(assign.values, ~[1u, 1u, 1u]);
		assign.next();
		assert_eq!(assign.values, ~[0u, 0u, 0u]);
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
	fn test_vars_cardinality() {
		let table = get_symb_table_1();
		let vs1 = ~[1];
		let vs2 = ~[0, 2];
		let vs3 = ~[0, 1, 2];

		assert_eq!(table.vars_cardinality(vs1), 2);
		assert_eq!(table.vars_cardinality(vs2), 4);
		assert_eq!(table.vars_cardinality(vs3), 8);
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

	#[test]
	fn test_union_vars() {
		let table = get_symb_table_1();
		let f1 = Factor::new(~[0, 1], &table,
			                 ~[0.25, 0.25, 0.25, 0.25]);
		let f2 = Factor::new(~[1, 2], &table,
			                 ~[0.25, 0.25, 0.25, 0.25]);
		let f3 = Factor::new(~[0, 1, 2], &table,
			                 ~[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]);
		let f4 = Factor::new(~[1], &table,
			                 ~[0.25, 0.25]);

		assert_eq!(f1.union_vars(&f2), ~[0, 1, 2]);
		assert_eq!(f2.union_vars(&f1), ~[1, 2, 0]); // implementation is order-sensitive
		assert_eq!(f1.union_vars(&f3), ~[0, 1, 2]);
		assert_eq!(f2.union_vars(&f3), ~[1, 2, 0]);
		assert_eq!(f1.union_vars(&f4), ~[0, 1]);
		assert_eq!(f2.union_vars(&f4), ~[1, 2]);
	}

    #[test]
	fn test_union_vars_many() {
		let table = get_symb_table_1();
		let f1 = Factor::new(~[0], &table,
			                 ~[0.25, 0.25]);
		let f2 = Factor::new(~[2], &table,
			                 ~[0.25, 0.25]);
		let f3 = Factor::new(~[0, 1], &table,
			                 ~[0.25, 0.25, 0.25, 0.25]);
		let f4 = Factor::new(~[1], &table,
			                 ~[0.25, 0.25]);

		assert_eq!(f1.union_vars_many([&f2, &f4]), ~[0, 2, 1]);
		assert_eq!(f1.union_vars_many([&f4, &f2]), ~[0, 1, 2]);
		assert_eq!(f2.union_vars_many([&f1, &f4]), ~[2, 0, 1]);
		assert_eq!(f3.union_vars_many([&f2]), ~[0, 1, 2]);
		assert_eq!(f3.union_vars_many([&f2, &f4]), ~[0, 1, 2]);
		assert_eq!(f1.union_vars_many([&f3, &f4, &f2]), ~[0, 1, 2]);
	}

}
