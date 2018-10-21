cc_binary(
    name = "main_ptrs",
    srcs = ["main_ptrs.cc"],
    deps = [
#        ":hello-greet",
        "//lib:ptrs_unique_ptr_util",
        "//lib:ptrs_shared_ptr_util",
    ],
    linkopts = [
        "-std=c++11",
    ], 
)

cc_binary(
    name = "main_exceptions",
    srcs = ["main_exceptions.cc"],
    deps = [
#        ":hello-greet",
        "//lib:exceptions_exceptions",
    ],
    linkopts = [
        "-std=c++11",
    ], 
)
