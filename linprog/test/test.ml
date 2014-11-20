open OUnit2

let test_read_dict ctxt = 
  let f = open_in "test/part1/dict1" in
  let d = Dict.read f in
  assert_equal d.m 3;
  assert_equal d.n 4;
  assert_equal (Array.length d.basis) d.m;
  assert_equal d.basis.(0) 1


let suite = 
  "Unit tests" >:::
    [ "dictionary reading" >:: test_read_dict ]

let () = 
  run_test_tt_main suite
