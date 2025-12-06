graph [
  directed 1
  node [
    id 0
    label "nodes_alloc"
  ]
  node [
    id 1
    label "node_hours"
  ]
  node [
    id 2
    label "runtime"
  ]
  node [
    id 3
    label "system_load"
  ]
  node [
    id 4
    label "cpus_alloc"
  ]
  node [
    id 5
    label "num_alloc_gpus"
  ]
  node [
    id 6
    label "mem_alloc"
  ]
  node [
    id 7
    label "status"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 0
    target 4
  ]
  edge [
    source 0
    target 6
  ]
  edge [
    source 2
    target 1
  ]
  edge [
    source 2
    target 3
  ]
  edge [
    source 3
    target 1
  ]
  edge [
    source 3
    target 2
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 4
    target 0
  ]
  edge [
    source 6
    target 5
  ]
  edge [
    source 6
    target 4
  ]
  edge [
    source 6
    target 0
  ]
  edge [
    source 7
    target 5
  ]
]
