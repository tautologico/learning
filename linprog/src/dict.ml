
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

let analyze_entering d = 
  Array.mapi (fun i x -> (x,i-1)) d.obj
  |> Array.to_list
  |> List.filter (fun (x,i) -> x > 0.0 && i >= 0)
  |> List.map (fun (_,i) -> d.nbasic.(i), i)
  |> Util.lst_foldl1 (fun (x1,i1) (x2,i2) -> if x2 < x1 then (x2,i2) else (x1,i1))
  |> Util.lift_option snd 

let find_min_pos_index dict v = 
  let select_pair (i1,x1) (i2,x2) = 
    if x1 < x2 then (i1,x1) 
    else if x1 <= x2 && i1 < i2 then (i1,x1)
    else (i2,x2)
  in
  List.mapi (fun i x -> (i, x)) v 
  |> List.filter (fun (_,x) -> x >= 0.0)
  |> List.map (fun (i,x) -> (dict.basic.(i), x))
  |> Util.lst_foldl1 select_pair 

let analyze_leaving d enter = 
  Array.to_list d.assign 
  |> List.mapi (fun i x -> -. x /. (Matrix.get d.a i enter))
  |> find_min_pos_index d



  
  
