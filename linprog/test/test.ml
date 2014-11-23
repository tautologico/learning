open OUnit2

let test_read_dict ctxt = 
  let f = open_in "test/part1/dict1" in
  let d = Dict.read f in
  let open Dict in 
  assert_equal ~msg:"m should be 3" d.m 3;
  assert_equal ~msg:"n should be 4" d.n 4;
  assert_equal ~msg:"length" (Array.length d.basic) d.m;
  assert_equal ~msg:"1st basic var" d.basic.(0) 1;
  assert_equal ~msg:"2nd basic var" d.basic.(1) 3;
  assert_equal ~msg:"3rd basic var" d.basic.(2) 6;
  assert_equal ~msg:"1st nonbasic var" d.nbasic.(0) 2;
  assert_equal ~msg:"2nd nonbasic var" d.nbasic.(1) 4;
  assert_equal ~msg:"3rd nonbasic var" d.nbasic.(2) 5;
  assert_equal ~msg:"4th nonbasic var" d.nbasic.(3) 7


let suite = 
  "Unit tests" >:::
    [ "dictionary reading" >:: test_read_dict ]

let () = 
  run_test_tt_main suite
