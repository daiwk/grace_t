/*Copyright 2018 The grace_t Authors. All Rights Reserved.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <iostream>
#include "lib/ptrs/unique_ptr_util.h"


int func_unique_ptr() {
    //// unique_ptr
    std::unique_ptr<int> up0(new int());
    //std::unique_ptr<int> up2 = new int();   //error! 构造函数是explicit
    //std::unique_ptr<int> up3(up1); ////error! 不允许拷贝

    std::unique_ptr<int> up(new int(10));
    // 传引用，不拷贝，不涉及所有权的转移
    grace_t::lib::ptrs::func3(up);
    // 暂时转移所有权，函数结束时返回拷贝，重新收回所有权
    // 如果不用up重新接受func2的返回值，这块内存就泄漏了
    up = grace_t::lib::ptrs::func4(std::unique_ptr<int> (up.release()));
    // up放弃对它所指对象的控制权，并返回保存的指针，将up置为空，不会释放内存
    up.release();

    //释放up指向的对象，将up置为空== up.reset();
    up = nullptr;

    int *x(new int());
    std::unique_ptr<int> up1,up2;
    // up.reset(…) 参数可以为 空、内置指针，先将up所指对象释放，然后重置up的值.
    up1.reset(x);
    // 不能再下面这么做了，因为会报：pointer being freed was not allocated
    // 因为这样会使up1 up2指向同一个内存，但unique_ptr不允许两个独占指针指向同一个对象
    //up2.reset(x);
    return 0;
}


int main()
{
    func_unique_ptr();
    return 0;
}

