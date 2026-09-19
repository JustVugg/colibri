#include "gemma4_tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    static const char tools[] =
        "[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\","
        "\"description\":\"Get current weather.\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"unit\":{\"type\":\"string\","
        "\"enum\":[\"celsius\",\"fahrenheit\"]},"
        "\"city\":{\"type\":\"string\",\"description\":\"City name\"}},"
        "\"required\":[\"city\"]}}}]";
    static const char expected[] =
        "<|tool>declaration:get_weather{description:<|\"|>Get current weather."
        "<|\"|>,parameters:{properties:{city:{description:<|\"|>City name"
        "<|\"|>,type:<|\"|>STRING<|\"|>},unit:{enum:[<|\"|>celsius"
        "<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},"
        "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
    static const char nested_tools[] =
        "[{\"type\":\"function\",\"function\":{\"name\":\"plan\","
        "\"description\":\"\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"days\":{\"type\":\"array\",\"items\":{"
        "\"type\":\"object\",\"properties\":{\"date\":{"
        "\"type\":\"string\"}},\"required\":[\"date\"]}},"
        "\"options\":{\"type\":\"object\",\"properties\":{\"detail\":{"
        "\"type\":\"boolean\",\"nullable\":true}},\"required\":[]}},"
        "\"required\":[]}}}]";
    static const char nested_expected[] =
        "<|tool>declaration:plan{description:<|\"|><|\"|>,parameters:{"
        "properties:{days:{items:{properties:{date:{type:<|\"|>STRING"
        "<|\"|>}},required:[<|\"|>date<|\"|>],type:<|\"|>OBJECT<|\"|>},"
        "type:<|\"|>ARRAY<|\"|>},options:{properties:{detail:{nullable:true,"
        "type:<|\"|>BOOLEAN<|\"|>}},type:<|\"|>OBJECT<|\"|>}},type:<|\"|>"
        "OBJECT<|\"|>}}<tool|>";
    char error[COLI_GEMMA4_TOOLS_ERROR_MAX];
    coli_gemma4_tool_call calls[2];
    char *rendered = NULL;
    size_t rendered_bytes = 0, call_count = 0;
    int failed = 0;
    if (coli_gemma4_render_tools(tools, sizeof(tools) - 1,
                                 &rendered, &rendered_bytes,
                                 error, sizeof(error)) != 0 ||
        rendered_bytes != sizeof(expected) - 1 ||
        memcmp(rendered, expected, sizeof(expected) - 1) != 0) {
        fprintf(stderr, "tool render mismatch: %s\n", error);
        if (rendered) fprintf(stderr, "actual: %s\n", rendered);
        failed = 1;
    }
    free(rendered);
    rendered = NULL;
    if (!failed &&
        (coli_gemma4_render_tools(
             nested_tools, sizeof(nested_tools) - 1,
             &rendered, &rendered_bytes, error, sizeof(error)) != 0 ||
         rendered_bytes != sizeof(nested_expected) - 1 ||
         memcmp(rendered, nested_expected, sizeof(nested_expected) - 1) != 0)) {
        fprintf(stderr, "nested tool render mismatch: %s\n", error);
        if (rendered) fprintf(stderr, "actual: %s\n", rendered);
        failed = 1;
    }
    free(rendered);
    rendered = NULL;
    if (!failed) {
        static const char generated[] =
            "note<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>}"
            "<tool_call|><|tool_call>call:get_time{}<tool_call|>"
            "<|tool_response>";
        if (coli_gemma4_parse_tool_calls(
                generated, sizeof(generated) - 1, calls, 2, &call_count,
                error, sizeof(error)) != 0 || call_count != 2 ||
            strcmp(calls[0].name, "get_weather") ||
            strcmp(calls[1].name, "get_time") ||
            calls[0].arguments_bytes !=
                strlen("{city:<|\"|>Rome<|\"|>}") ||
            memcmp(generated + calls[0].arguments_offset,
                   "{city:<|\"|>Rome<|\"|>}",
                   calls[0].arguments_bytes) != 0) {
            fprintf(stderr, "tool call parse mismatch: %s\n", error);
            failed = 1;
        }
    }
    if (!failed) {
        static const char response_json[] =
            "{\"temp_c\":31,\"city\":\"Rome\",\"conditions\":\"sunny\"}";
        static const char response_expected[] =
            "response:get_weather{city:<|\"|>Rome<|\"|>,conditions:<|\"|>"
            "sunny<|\"|>,temp_c:31}<tool_response|>";
        if (coli_gemma4_render_tool_response(
                "get_weather", response_json, sizeof(response_json) - 1, 0,
                &rendered, &rendered_bytes, error, sizeof(error)) != 0 ||
            rendered_bytes != sizeof(response_expected) - 1 ||
            memcmp(rendered, response_expected,
                   sizeof(response_expected) - 1) != 0) {
            fprintf(stderr, "tool response render mismatch: %s\n", error);
            if (rendered) fprintf(stderr, "actual: %s\n", rendered);
            failed = 1;
        }
        free(rendered);
        rendered = NULL;
    }
    if (!failed) {
        static const char scalar_expected[] =
            "<|tool_response>response:get_time{value:<|\"|>noon<|\"|>}"
            "<tool_response|>";
        if (coli_gemma4_render_tool_response(
                "get_time", "\"noon\"", 6, 1,
                &rendered, &rendered_bytes, error, sizeof(error)) != 0 ||
            rendered_bytes != sizeof(scalar_expected) - 1 ||
            memcmp(rendered, scalar_expected,
                   sizeof(scalar_expected) - 1) != 0) {
            fprintf(stderr, "scalar tool response mismatch: %s\n", error);
            failed = 1;
        }
        free(rendered);
        rendered = NULL;
    }
    if (!failed && coli_gemma4_render_tool_response(
            "get_time", "\"unterminated", 13, 0,
            &rendered, &rendered_bytes, error, sizeof(error)) == 0) {
        fprintf(stderr, "unterminated tool response JSON was accepted\n");
        failed = 1;
    }
    free(rendered);
    rendered = NULL;
    if (!failed &&
        (coli_gemma4_render_tools("[]", 2, &rendered, &rendered_bytes,
                                  error, sizeof(error)) != 0 ||
         rendered_bytes != 0 || !rendered || rendered[0])) failed = 1;
    free(rendered);
    rendered = NULL;
    if (!failed && coli_gemma4_render_tools(
            "{}", 2, &rendered, &rendered_bytes,
            error, sizeof(error)) == 0) failed = 1;
    free(rendered);
    if (!failed) puts("Gemma 4 tool renderer tests passed");
    return failed;
}
