/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file ./main/hello-world.cc
 * @author wkdai(wkdai@baidu.com)
 * @date 2018/10/20 20:40:27
 * @version $Revision$ 
 * @brief 
 *  
 **/
#include <iostream>
#include "lib/ptrs/unique_ptr_util.h"

int main()
{

    std::unique_ptr<int> up(new int(10));
    grace_t::ptrs::func3(up);
    return 0;
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
