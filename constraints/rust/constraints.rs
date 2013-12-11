
type Var = uint;
type Value = uint;

struct FiniteDomain {
	values: ~[Value]
}

impl FiniteDomain {
	pub fn remove(&mut self, val: Value) {
		match self.values.position_elem(&val) {
			Some(ix) => { self.values.remove(ix); }
			None => ()
		}
	}
}

struct FDConstraint {
	scope:    ~[Var],
	relation: ~[Value]  // 2-dimensional values matrix
}

struct FDConstraintNetwork {
	nvars:       uint,
	domains:     ~[FiniteDomain],
	constraints: ~[FDConstraint]
}

impl FDConstraintNetwork {
	pub fn revise(&mut self, vi: Var, vj: Var, cons: &FDConstraint) {
		println("revise")
	}
}

#[test]
fn some_test() {
	println("Wumpagaramba")
}

#[cfg(test)]
mod tests {
	use super::FiniteDomain;

	#[test]
	fn test_remove() {
		let mut fd1 = FiniteDomain { values: ~[0, 1, 2, 3] };
		fd1.remove(2);
		assert_eq!(fd1.values, ~[0, 1, 3]);
	}
}
