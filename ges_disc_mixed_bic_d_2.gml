graph [
  directed 1
  node [
    id 0
    label "nodes_alloc_cat"
  ]
  node [
    id 1
    label "cpus_alloc_cat"
  ]
  node [
    id 2
    label "gpu_cat"
  ]
  node [
    id 3
    label "mem_alloc_cat"
  ]
  node [
    id 4
    label "runtime_bin"
  ]
  node [
    id 5
    label "node_hours_bin"
  ]
  node [
    id 6
    label "status"
  ]
  node [
    id 7
    label "system_load_cat"
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 1
    target 6
  ]
  edge [
    source 1
    target 3
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 1
    target 4
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 2
    target 6
  ]
  edge [
    source 2
    target 4
  ]
  edge [
    source 2
    target 0
  ]
  edge [
    source 3
    target 6
  ]
  edge [
    source 3
    target 2
  ]
  edge [
    source 3
    target 0
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 3
    target 5
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 4
    target 7
  ]
  edge [
    source 5
    target 7
  ]
]
