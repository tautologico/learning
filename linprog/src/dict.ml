
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
  let basic = Util.read_int_array ch m in
  let nbasic = Util.read_int_array ch n in
  let assign = Util.read_vector ch m in
  let a = Matrix.read ch ~rows:m ~cols:n in
  let obj = Util.read_vector ch (n+1) in
  { m; n; basic; nbasic; assign; a; obj }


