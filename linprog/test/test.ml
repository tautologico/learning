open OUnit2

let test_foldl1 ctxt = 
  assert_equal (Util.lst_foldl1 max [8; 11; 3; 2; 0; 1; 3]) (Some 11)

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

let test_entering ctxt = 
  let d = Dict.read_file "test/part1/dict1" in
  assert_equal (Dict.analyze_entering d) (Some 1);
  let d5 = Dict.read_file "test/part1/dict5" in
  assert_equal (Dict.analyze_entering d5) (Some 1);
  let d6 = Dict.read_file "test/part1/dict6" in
  assert_equal (Dict.analyze_entering d6) (Some 0)

let test_find_min_pos ctxt = 
  let open Dict in
  let d = { m = 6; n = 6; basic = [| 3; 5; 4; 2; 1; 7 |]; nbasic = [||];
            assign = [||]; obj = [||]; a = Matrix.zero ~rows:6 ~cols:6 } in
  let v1 = [1.2; 0.015; -2.3; 4.7; 0.2; 4.4] in
  assert_equal (find_min_pos_index d v1) (Some (1, 0.015));
  let v2 = [-1.3; -1.5; -1.1; -5.5; -4.4; -1.23] in
  assert_equal (find_min_pos_index d v2) None;
  let v3 = [-1.2; 0.003; 4.4; 0.15; 0.003; 6.7] in
  assert_equal (find_min_pos_index d v3) (Some (4, 0.003));
  let v4 = [-1.2; 0.003; 4.4; 0.15; 1.003; 0.003] in
  assert_equal (find_min_pos_index d v4) (Some (1, 0.003))

let test_leaving ctxt = 
  let d = Dict.read_file "test/part1/dict1" in
  assert_equal (Dict.analyze_entering d) (Some 1);
  assert_equal (Dict.analyze_leaving d 1) (Some (1, 3.0));
  let d5 = Dict.read_file "test/part1/dict5" in
  assert_equal (Dict.analyze_entering d5) (Some 1); 
  let d6 = Dict.read_file "test/part1/dict6" in
  assert_equal (Dict.analyze_entering d6) (Some 0)

let test_read_solution ctxt = 
  let cmp = Util.lift_eq_option Dict.eq_solution in 
  let open Dict in 
  let sol1 = Dict.read_solution_file "test/part2/dict1.output" in
  assert_equal ~cmp sol1 (Some { steps = 3; final_val = 7.0 });
  let sol2 = Dict.read_solution_file "test/part2/dict2.output" in
  assert_equal ~cmp sol2 (Some { steps = 1; final_val = 4.0 });
  let sol5 = Dict.read_solution_file "test/part2/dict5.output" in
  assert_equal ~cmp sol5 (Some { steps = 4; final_val = 60.0 });
  let sol6 = Dict.read_solution_file "test/part2/dict6.output" in
  assert_equal sol6 None

let suite = 
  "Unit tests" >:::
    [ 
      "foldl1" >:: test_foldl1;
      "dictionary reading" >:: test_read_dict;
      "find_min_pos" >:: test_find_min_pos;
      "entering var analysis" >:: test_entering;
      "leaving var analysis" >:: test_leaving;
      "reading expected solution" >:: test_read_solution
    ]

let test_pivot_part1 i ctxt = 
  let dict_file = Printf.sprintf "test/part1/dict%d" i in
  let ps_file = Printf.sprintf "test/part1/dict%d.output" i in 
  let dict = Dict.read_file dict_file in
  let exp_ps = Dict.read_pivot_step_file ps_file in
  let calc_ps = Dict.calc_pivot_step dict in
  assert_equal ~cmp:(Util.lift_eq_option Dict.eq_pivot_step) exp_ps calc_ps

let pivot_part1_suite = 
  let test_name i = Printf.sprintf "dict%d" i in
  let suite_list = 
    List.map 
      (fun i -> (test_name i) >:: (test_pivot_part1 i)) 
      (Util.range_lst 1 10)
  in
  "pivot test, part 1" >::: suite_list

let () = 
  Printf.printf "# Unit tests:\n";
  run_test_tt_main suite;
  Printf.printf "\n# pivoting tests, part 1:\n";
  run_test_tt_main pivot_part1_suite
