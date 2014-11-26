type t = {
  m: int;
  n: int;
  basic: int array;
  nbasic: int array;
  assign: float array;
  a : Matrix.t;
  obj : float array;
}

val read : in_channel -> t 

val find_min_pos_index : t -> float list -> (int * float) option

val analyze_entering : t -> int option 

val analyze_leaving : t -> int -> (int * float) option 

