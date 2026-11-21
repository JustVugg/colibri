#include "../coli_sycl_matmul_plan.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <initializer_list>
int main() {
    assert(coli_sycl_mm_rows(22,28672)==22);
    assert(coli_sycl_mm_rows(338,28672)==292);
    assert((int64_t)292*28672*256<=INT_MAX);
    assert((int64_t)293*28672*256>INT_MAX);
    assert(coli_sycl_mm_rows(338,28672,1000)==292);
    assert(coli_sycl_mm_rows(7,3,3)==3);
    assert(coli_sycl_mm_rows(7,3,0)==7);
    assert(coli_sycl_mm_rows(INT_MAX,1)==INT_MAX/256);
    assert(coli_sycl_mm_rows(1,INT_MAX)==0);
    assert(coli_sycl_mm_rows(0,3)==0 && coli_sycl_mm_rows(3,0)==0);
    for(const char *s:{"auto","-1","garbage","12x","999999999999999999999"})
        assert(coli_sycl_mm_override(s)==-1);
    assert(coli_sycl_mm_override(nullptr)==-1);
    assert(coli_sycl_mm_override("0")==0 && coli_sycl_mm_override("3")==3);
    std::puts("MATMUL PLAN PASS: native overflow boundary, tiny forced chunks, invalid/legacy controls");
}
