
type t = {
  m: int;
  n: int;
  basic: int array;
  nbasic: int array;
  assign: float array;
  a : Matrix.t;
  obj : float array;
}

let read ch = 
  let m = Util.read_int ch in
  let n = Util.read_int ch in
  { m; n;
    basic = Util.read_int_array ch m;
    nbasic = Util.read_int_array ch n;
    assign = Util.read_vector ch m;
    a = Matrix.read ch ~rows:m ~cols:n;
    obj = Util.read_vector ch (n+1) }


