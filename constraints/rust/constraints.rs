
type Var = uint;
type Value = uint;

struct Queue<A> {
	queue: ~[A],
	head: uint,
	tail: uint
}

impl<A: Default + Clone> Queue<A> {
	pub fn new(size: uint) -> Queue<A> {
		let mut q = std::vec::with_capacity(size);
		for _i in range(0, size) {
			q.push(Default::default());
		}
		Queue { queue: q, head: 0, tail: 0 }
	}

	pub fn add(&mut self, v: A) {
		self.queue[self.tail] = v;
		self.tail += 1;
	}

	pub fn remove_front(&mut self) -> A {
		let res = self.queue[self.head].clone();
		self.head += 1;
		res
	}
}

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

struct BinaryFDConstraint {
	scope:     (Var, Var),
	relation: ~[(Value, Value)]
}

impl BinaryFDConstraint {
	fn contains_v1(&self, v1: Value) -> bool {
		self.relation.iter().any(|&(v, _)| v == v1)
	}

	fn contains_v2(&self, v2: Value) -> bool {
		self.relation.iter().any(|&(_, v)| v == v2)
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

struct BinaryFDConstraintNetwork {
	nvars:       uint,
	domains:     ~[FiniteDomain],
	constraints: ~[BinaryFDConstraint]
}

impl BinaryFDConstraintNetwork {
	pub fn revise(&mut self, cix: uint, vi: Var) {
		let (v1, v2) = self.constraints[cix].scope;
		assert!(v1 == vi || v2 == vi);
		let is_first = (vi == v1);
		let mut removals = std::vec::with_capacity(self.domains[vi].values.len());

		for v in self.domains[vi].values.iter() {
			if (is_first && !self.constraints[cix].contains_v1(*v)) || 
			   (!is_first && !self.constraints[cix].contains_v2(*v)) {
				removals.push(*v);
			}
		}

		for v in removals.iter() {
			self.domains[vi].remove(*v);
		}
	}

	pub fn ac3(&mut self) {
		println("Consistency")
	}
}


#[cfg(test)]
mod tests {
	use super::FiniteDomain;
	use super::BinaryFDConstraint;
	use super::BinaryFDConstraintNetwork;

	#[test]
	fn test_remove() {
		let mut fd1 = FiniteDomain { values: ~[0, 1, 2, 3] };
		fd1.remove(2);
		assert_eq!(fd1.values, ~[0, 1, 3]);
		fd1.remove(1);
		assert_eq!(fd1.values, ~[0, 3]);
		fd1.remove(0);
		assert_eq!(fd1.values, ~[3]);
		fd1.remove(3);
		assert_eq!(fd1.values, ~[]);
	}

	#[test]
	fn test_revise() {
		let domains = ~[FiniteDomain { values: ~[1, 2, 3] }, FiniteDomain { values: ~[1, 2, 3] }];
		let constraint = BinaryFDConstraint { scope: (0, 1), relation: ~[(1, 1), (2, 1)] };
		let mut cnet = BinaryFDConstraintNetwork { nvars: 2, domains: domains, 
			                                       constraints: ~[constraint] };
        
        cnet.revise(0, 0);
        assert_eq!(cnet.domains[0].values, ~[1, 2]);
        cnet.revise(0, 1);
        assert_eq!(cnet.domains[1].values, ~[1]);
	}
}
