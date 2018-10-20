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

#include <memory>

std::unique_ptr<int> up1(new int());

////从函数返回一个unique_ptr
//std::unique_ptr func1(int a)
//{
//    return std::unique_ptr<int> (new int(a));
//}
 
////返回一个局部对象的拷贝
//unique_ptr func2(int a)
//{
//    unique_ptr<int> up(new int(a));
//    return up;
//}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
