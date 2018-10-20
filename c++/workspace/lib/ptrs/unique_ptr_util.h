/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file lib/ptrs/unique_ptr_util.h
 * @author wkdai(wkdai@baidu.com)
 * @date 2018/10/20 20:37:08
 * @version $Revision$ 
 * @brief 
 *  
 **/

// refers to https://blog.csdn.net/qq_33266987/article/details/78784286

#include <memory>
#include <iostream>

namespace grace_t {
namespace ptrs {


//从函数返回一个unique_ptr
std::unique_ptr<int> func1(int a);
 
//返回一个局部对象的拷贝
std::unique_ptr<int> func2(int a);

void func3(std::unique_ptr<int> &up);

std::unique_ptr<int> func4(std::unique_ptr<int> up);

}
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
