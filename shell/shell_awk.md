#AWK概述
AWK是一种解释型编程语言。它非常强大，专为**文本处理和设计**。
#AWK-工作流程
AWK遵循简单的工作流程-**读取、执行和重复**

![AWK流程](https://www.w3schools.cn/awk/images/awk_workflow.jpg)

> 读取

> AWK 从输入流（文件、管道或标准输入）中读取一行并将其存储在内存中。
> 
> 执行
> 
> 所有 AWK 命令都按顺序应用于输入。 默认情况下，AWK 在每一行上执行命令。 我们可以通过提供模式来限制这一点。
> 
> 重复
> 
> 这个过程会一直重复，直到文件结束。

#AWK-基本语法
    awk [options] file ...
    awk '{print}' marks.txt #使用 AWK 显示文件的完整内容
    awk [options] -f file .... #可以在脚本文件中提供 AWK 命令
    awk -f command.awk marks.txt 

**AWK标准选项**

-v选项

此选项为变量赋值。它允许在程序执行之前进行赋值。

    awk -v name=Jerry 'BEGIN{printf "Name = %s\n", name}'
    输出：Name = Jerry

--dump-variables[=file] 选项

它将全局变量及其最终值的排序列表打印到文件中。 默认文件是 awkvars.out。

示例

    [jerry]$ awk --dump-variables ''
    [jerry]$ cat awkvars.out 

--lint[=fatal] 选项

此选项可以检查不可移植或可疑的构造。 当提供参数 fatal 时，它会将警告消息视为错误。 以下示例演示了这一点 −

示例

    [jerry]$ awk --lint '' /bin/ls
    
    输出
    awk: cmd. line:1: warning: empty program text on command line
    awk: cmd. line:1: warning: source file does not end in newline
    awk: warning: no program text at all!

--posix 选项

此选项打开严格的 POSIX 兼容性，其中所有常见的和 gawk 特定的扩展都被禁用。

--profile[=file] 选项
此选项在文件中生成程序的漂亮打印版本。 默认文件是 awkprof.out。 下面的简单示例说明了这一点 −

示例

    [jerry]$ awk --profile 'BEGIN{printf"---|Header|--\n"} {print} 
    END{printf"---|Footer|---\n"}' marks.txt > /dev/null 
    [jerry]$ cat awkprof.out
在执行此代码时，您会得到以下结果 −

输出

    # gawk profile, created Sun Oct 26 19:50:48 2014
    
       # BEGIN block(s)
    
       BEGIN {
      printf "---|Header|--\n"
       }
    
       # Rule(s) {
      print $0
       }
    
       # END block(s)
    
       END {
      printf "---|Footer|---\n"
       }
--traditional 选项

此选项禁用所有 gawk 特定的扩展。

--version 选项

此选项显示 AWK 程序的版本信息。

示例

    [jerry]$ awk --version
    输出
    GNU Awk 4.0.1
    Copyright (C) 1989, 1991-2012 Free Software Foundation.
