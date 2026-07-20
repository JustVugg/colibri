package main

import (
	"os/exec"
	"strconv"
	"strings"
)

func cmdRun(a *Args) int {
	needModel(a.Model)
	if len(a.Prompt) == 0 {
		fatal(`usage: coli run "your prompt"`)
	}
	prompt := strings.Join(a.Prompt, " ")
	banner("run")
	// Official GLM-5.2 template: no \n after the roles; <think></think> = direct
	// answer (nothink).
	e := envFor(a)
	e["PROMPT"] = "[gMASK]<sop><|user|>" + prompt + "<|assistant|><think></think>"
	cmd := exec.Command(GLM, strconv.Itoa(a.Cap))
	cmd.Env = envToSlice(e)
	return runInherit(cmd)
}
