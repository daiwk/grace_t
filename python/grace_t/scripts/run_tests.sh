#!/bin/bash

function test_keras()
{
    python keras_demo.py
    return $?
}

function test_xgboost()
{
    python xgboost_demo.py
    return $?
}

function main()
{
    test_xgboost
    [[ $? -ne 0 ]] && exit 1
    test_keras
    [[ $? -ne 0 ]] && exit 1
    return 0
}
main 2>&1 
