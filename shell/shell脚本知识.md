#一、shell中`#*,##*,#*,##*,% *,%% *`的含义及用法
假设定义了一个变量为：
代码如下: `file=/dir1/dir2/dir3/my.file.txt`

可以用${ }分别替换得到不同的值：

* `${file#*/}`：删掉第一个 / 及其左边的字符串：`dir1/dir2/dir3/my.file.txt`
* `${file##*/}`：删掉最后一个 / 及其左边的字符串：`my.file.txt`
* `${file#*.}`：删掉第一个 . 及其左边的字符串：`file.txt`
* `${file##*.}`：删掉最后一个 . 及其左边的字符串：`txt`
* `${file%/*}`：删掉最后一个 / 及其右边的字符串：`/dir1/dir2/dir3`
* `${file%%/*}`：删掉第一个 / 及其右边的字符串：(空值)
* `${file%.*}`：删掉最后一个 . 及其右边的字符串：`/dir1/dir2/dir3/my.file`
* `${file%%.*}`：删掉第一个 . 及其右边的字符串：`/dir1/dir2/dir3/my`


记忆的方法为：

* #是去掉左边（键盘上#在 $ 的左边）
* %是去掉右边（键盘上% 在$ 的右边）
* 单一符号是最小匹配；两个符号是最大匹配

#二、shell中各种括号的作用详解()、(())、[]、[[]]、{}
字符串的运算符：

* `=`	检测两个字符串是否相等，相等返回 true。  例如:	`[ $a = $b ] `返回 false。
* `!=`	检测两个字符串是否相等，不相等返回 true。 例如：`[ $a != $b ]` 返回 true。
* `-z`	检测字符串长度是否为0，为0返回 true。 例如： `[ -z $a ]` 返回 false。
* `-n`	检测字符串长度是否不为 0，不为 0 返回 true。 例如：`[ -n "$a" ]` 返回 true。
* `$`	检测字符串是否为空，不为空返回 true。	`[ $a ]` 返回 true。
* `-d `  判断目录是否存在
* `-e `filename 判断是否存在文件
* `-f` file 判断是不是文件

##2.1小括号()
小括号可以用来定义一个数组变量，如下：

    array1=(1 2 3 4 5)
    array2=(one two three four five)
打印结果，使用${}替换：

    echo ${array1}  # 1 
    echo ${array1[2]} # 3
    echo ${array2[3]} # four
> 注意：在 shell 中使用数组变量有时会引起一些问题，而且数组变量的可移植性并不好，因此在 shell 编程中，数组变量使用得并不多。
##2.2双小括号(())
双小括号命令允许在比较过程中使用高级数学表达式： ((expression))

双小括号命令提供了更多的数学符号，可以在双小括号中进行各种逻辑运算、数学运算，也支持更多的运算符（如 ++、-- 等）。常使用的双小括号来在 for 循环中实现 C 语言风格的迭代, 代码如下：

    for ((i=0; i<10; i++))
    do
       echo -n "$i "
    done
    echo " "
##2.3中括号 []
单个的中括号的功能与 test 命令一样，都是用作条件测试。
    read -p "please enter a number: " num
    if [$num -gt 10];then
       echo "num > 10"
    else
       echo "num <= 10"
    fi
##2.4双中括号[[]]
双中括号提供了针对字符串比较的高级特性，使用双中括号 `[[ ]] `进行字符串比较时，可以把右边的项看做一个模式，故而可以在 `[[ ]]` 中使用正则表达式：

    if [[ hello == hell* ]];then
    echo "equal"
    else
     echo "uneuqal"
    fi 
    exit 0
##2.5大括号 { }
大括号用于括起一个语句块。如果需要在某些只能使用单个语句的地方（如AND、OR列表中）使用多条语句，则可以用大括号将这多条语句括起来构造一个语句块。

##2.6相关字符总结
* 双引号 `""`, 引用字符串，字符串中部分特殊符合有意义
* 单引号 `''`, 引用字符串，字符串中特殊符合全都没有意义
* 反引号``、$() , 命令替换
* `$(())、$[]、(())`, 算术运算
* `${}`,  变量替换
* (),  数组初始化
* [],    条件测试
* [[]],   字符串比较
* {},    括起一个语句块

多个条件判断，[] 和 [[]] 的区别？
> [[ ]] 双对中括号，是不能使用 -a 或者 -o的参数进行比较的。
> 
> &&:并且 ||:或 -a:并且 -o:或者。
> 
> [ ] 可以使用 -a -o的参数，但是必须在 [ ] 中括号内，判断条件
> 
> 当判断某个变量的值是否满足正则表达式的时候，必须使用[[ ]] 双对中括号

#三、重定向符合、管道符号、其他符号
##3.1重定向符号
>在shell脚本中有两种常见的重定向符号` >` 和 `>>`

 `>`符号的作用，其表示将符号左侧的内容，以覆盖的方式输入到右侧文件中

    echo 'world' > file.txt
    cat file.txt
    输出：world

 `>>`符号的作用表示将符号左侧的内容，以追加的方式输入到右侧文件的末尾行中，例如：

	echo '123' > file.txt
    cat file.txt
    输出：world  123
##3.2管道符号|
>定义： |这个就是管道符号，传递信息使用的。

>使用格式：
>命令1|命令2
>
>管道符左侧命令1执行后的结果，传递给管道符右侧的命令2使用。

>命令演示：查看当前系统中的全局变量。
>
    env | grep SHELL
    SHELL=/bin/bash
##3.3其他符号
###3.3.1后台展示符号 & 
>定义：&就是将一个命令从前台转到后台执行。
>
>使用格式：命令 &


    sleep 4 #此时前台操作，会卡主界面，不能做任何操作
    sleep 4 & #后台执行，不影响操作可以通过查看进程查看
###3.3.2 全部信息符号 2>&1
符号详解：
> 1 表示正确输出的信息
> 
> 2 表示错误输出的信息
> 
> 2>&1 代表所有输出信息

符号示例：
>标准正确输出示例：cat nihao.txt 1 >> zhengque 
>
>标准错误输出示例：cat dadadda 2 >> errfile

演示：

	#脚本内容
    echo "下一条是错误命令"
    acaca
    bash ceshi.sh 1>> ceshi-ok 2>> ceshi-err #将正确的信息放到ceshi-ok里面，将错误的信息放到ceshi-err里面

    cat ceshi-ok
    输出：下一条是错误命令
    cat ceshi-err
    输出：ceshi.sh:行 acaca:未找到命令

    bash cashi.sh >> ceshi-all 2>&1  #全部信息
    cat ceshi-all
    输出：下一条是错误命令 ceshi.sh:行 acaca:未找到命令

###3.3.3 linux系统垃圾
> /dev/null是linux下的一个设备文件，这个文件类似于一个垃圾桶，特点是：容量无限大
	bash ceshi.sh >> /dev/null 2>&1 #直接扔进垃圾桶
#grep命令
>grep (global search regular expression(RE) and print out the line,全面搜索正则表达式并把行打印出来)是一种强大的文本搜索工具，它能使用正则表达式搜索文本，并把匹配的行打印出来。

grep常用用法：

	[root@www~]#grep [-acinv] [-color=auto] '搜寻字符串' filename

选项与参数：

    -a ：将 binary 文件以 text 文件的方式搜寻数据
    -c ：计算找到 '搜寻字符串' 的次数
    -i ：忽略大小写的不同，所以大小写视为相同
    -n ：输出行号
    -v ：反向选择，亦即显示出没有 '搜寻字符串' 内容的那一行！
    --color=auto ：可以将找到的关键词部分加上颜色的显示喔！
使用小总结

    grep -n 'the' regular_express.txt #搜索有the的行，并输出行号
    grep -nv 'the' regular_express.txt #搜索没有the的行，并输出行号
    #[]表示其中的某一字符，例如[ade]表示a或d或e
    grep -n 't[ae]st' regular_express.txt
    # 可以用^符号做[]内的前缀，表示除[]内的字符之外的字符
    grep -n '[^g]oo' regular_express.txt  #搜索ooq前没有g的字符串，
    grep -n '[0-9]' regular_express.txt #搜索包含数字的行
    # ^表示行的开头，$表示行的结尾，^$表示空行，因为只有行首和行尾
    grep -n '^the' regular_express.txt #搜索the在开头的行
    grep -n '^[a-z]' regular_express.txt #搜索以小写开头的行
    grep -v '^$' /etc/rsyslog.conf | grep -v '^#' #查询/etc/rsyslog.conf文件，但是不包含空行和注释行
##4.1 grep与正则表达式
(1) 字符类
字符类的搜索：如果我想要搜寻 test 或 taste 这两个单词时，可以发现到，其实她们有共通的 't?st' 存在～这个时候，我可以这样来搜寻：

    [root@www ~]# grep -n 't[ae]st' regular_express.txt
    
    8:I can't finish the test.
    
    9:Oh! The soup taste good.
(2) 字符类的反向选择 [^] ：如果想要搜索到有 oo 的行，但不想要 oo 前面有 g，如下:

	[root@www ~]# grep -n '[^g]oo' regular_express.txt
	
	2:apple is my favorite food.
	
	3:Football game is not use feet only.
	
	18:google is the best tools for search keyword.#此行是因为有tools
	
	19:goooooogle yes! #此行是因为有 ooo
(3) 字符类的连续：假设我 oo 前面不想要有小写字节，所以，我可以这样写 [^abcd....z]oo ， 但是这样似乎不怎么方便，由于小写字节的 ASCII 上编码的顺序是连续的， 因此，我们可以将之简化为底下这样

    [root@www ~]# grep -n '[^a-z]oo' regular_express.txt
    
    3:Football game is not use feet only.
(4)取得有数字的那一行：

    [root@www ~]# grep -n '[0-9]' regular_express.txt
    
    5:However, this dress is about $ 3183 dollars.
    
    15:You are the best is mean you are the no. 1.
(5) 行首与行尾字节 ^ $
行首字符：如果我想要让 the 只在行首列出呢？ 这个时候就得要使用定位字节了！

    [root@www ~]# grep -n '^the' regular_express.txt
    
    12:the symbol '*' is represented as start.
想要开头是小写字节的那一行就列出:

    [root@www ~]# grep -n '^[a-z]' regular_express.txt
    
    2:apple is my favorite food.
    
    4:this dress doesn't fit me.
    
    10:motorcycle is cheap than car.
    
    12:the symbol '*' is represented as start.
    
    18:google is the best tools for search keyword.
    
    19:goooooogle yes!
    
    20:go! go! Let's go.

不想要开头是英文字母，则可以是这样：

    [root@www ~]# grep -n '^[^a-zA-Z]' regular_express.txt
    
    1:"Open Source" is a good mechanism to develop programs.
    
    21:# I am VBird
找出来，行尾结束为小数点 (.) 的那一行:

    [root@www ~]# grep -n '\.$' regular_express.txt
    
    1:"Open Source" is a good mechanism to develop programs.
    
    2:apple is my favorite food.
    
    3:Football game is not use feet only.
    
    4:this dress doesn't fit me.
    
    10:motorcycle is cheap than car.
    
    11:This window is clear.
    
    12:the symbol '*' is represented as start.
找出空白行：

    [root@www ~]# grep -n '^$' regular_express.txt
    
    22:
(6)任意一个字节 . 与重复字节 *

    . (小数点)：代表『一定有一个任意字节』的意思；
    
    * (星号)：代表『重复前一个字符， 0 到无穷多次』的意思，为组合形态
找出 g??d 的字串，亦即共有四个字节， 起头是 g 而结束是 d ，可以这样做:

    [root@www ~]# grep -n 'g..d' regular_express.txt
    
    1:"Open Source" is a good mechanism to develop programs.
    
    9:Oh! The soup taste good.
    
    16:The world <Happy> is the same with "glad".
至少要有两个(含) o 以上，该如何是好。`『o*』`代表的是：『拥有空字节或一个 o 以上的字节』，因此，`『 grep -n 'o*' regular_express.txt 』`将会把所有的数据都列印出来终端上！当我们需要『至少两个 o 以上的字串』时，就需要` ooo* `，亦即是：

    [root@www ~]# grep -n 'ooo*' regular_express.txt
    
    1:"Open Source" is a good mechanism to develop programs.
    
    2:apple is my favorite food.
    
    3:Football game is not use feet only.
    
    9:Oh! The soup taste good.
    
    18:google is the best tools for search keyword.
    
    19:goooooogle yes!
我想要字串开头与结尾都是 g，但是两个 g 之间仅能存在至少一个 o ，亦即是 gog, goog, gooog.... 等等

    [root@www ~]# grep -n 'goo*g' regular_express.txt
    
    18:google is the best tools for search keyword.
    
    19:goooooogle yes!
找出 g 开头与 g 结尾的行，当中的字符可有可无：

    [root@www ~]# grep -n 'g.*g' regular_express.txt
    
    1:"Open Source" is a good mechanism to develop programs.
    
    14:The gd software is a library for drafting programs.
    
    18:google is the best tools for search keyword.
    
    19:goooooogle yes!
    
    20:go! go! Let's go.

如果我想要找出『任意数字』的行？因为仅有数字，所以就成为：

	[root@www ~]# grep -n '[0-9][0-9]*' regular_express.txt
	
	5:However, this dress is about $ 3183 dollars.
	
	15:You are the best is mean you are the no. 1.

限定连续 RE 字符范围 {}

我们可以利用 . 与 RE 字符及 * 来配置 0 个到无限多个重复字节， 那如果我想要限制一个范围区间内的重复字节数呢？举例来说，我想要找出两个到五个 o 的连续字串，该如何作？这时候就得要使用到限定范围的字符 {} 了。 但因为 { 与 } 的符号在 shell 是有特殊意义的，因此， 我们必须要使用字符   \ 来让他失去特殊意义才行。 至於 {} 的语法是这样的，假设我要找到两个 o 的字串，可以是：

	[root@www ~]# grep -n 'o\{2\}' regular_express.txt
	
	1:"Open Source" is a good mechanism to develop programs.
	
	2:apple is my favorite food.
	
	3:Football game is not use feet only.
	
	9:Oh! The soup taste good.
	
	18:google is the best tools for search ke
	
	19:goooooogle yes!

假设我们要找出 g 后面接 2 到 5 个 o ，然后再接一个 g 的字串，他会是这样：

	[root@www ~]# grep -n 'go\{2,5\}g' regular_express.txt
	
	18:google is the best tools for search keyword.

如果我想要的是 2 个 o 以上的 goooo....g 呢？除了可以是 gooo*g ，也可以是：

	[root@www ~]# grep -n 'go\{2,\}g' regular_express.txt
	
	18:google is the best tools for search keyword.
	
	19:goooooogle yes!

扩展grep(grep -E 或者 egrep)：

使用扩展grep的主要好处是增加了额外的正则表达式元字符集。

打印所有包含NW或EA的行。如果不是使用egrep，而是grep，将不会有结果查出。

	# egrep 'NW|EA' testfile
	
	northwest NW Charles Main 3.0 .98 3 34
	
	eastern EA TB Savage 4.4 .84 5 20

对于标准grep，如果在扩展元字符前面加\，grep会自动启用扩展选项-E。

	#grep 'NW\|EA' testfile
	
	northwest NW Charles Main 3.0 .98 3 34
	
	eastern EA TB Savage 4.4 .84 5 20

搜索所有包含一个或多个3的行。

	# egrep '3+' testfile
	
	# grep -E '3+' testfile
	
	# grep '3\+' testfile
	
	#这3条命令将会
	
	northwest NW Charles Main 3.0 .98 3 34
	
	western WE Sharon Gray 5.3 .97 5 23
	
	northeast NE AM Main Jr. 5.1 .94 3 13
	
	central CT Ann Stephens 5.7 .94 5 13

搜索所有包含0个或1个小数点字符的行。

	# egrep '2\.?[0-9]' testfile
	
	# grep -E '2\.?[0-9]' testfile
	
	# grep '2\.\?[0-9]' testfile
	
	#首先含有2字符，其后紧跟着0个或1个点，后面再是0和9之间的数字。
	
	western WE Sharon Gray 5.3 .97 5 23
	
	southwest SW Lewis Dalsass 2.7 .8 2 18
	
	eastern EA TB Savage 4.4 .84 5 20

搜索一个或者多个连续的no的行。

	# egrep '(no)+' testfile
	
	# grep -E '(no)+' testfile
	
	# grep 'nono\+' testfile #3个命令返回相同结果，
	
	northwest NW Charles Main 3.0 .98 3 34
	
	northeast NE AM Main Jr. 5.1 .94 3 13
	
	north NO Margot Weber 4.5 .89 5 9


不使用正则表达式

fgrep 查询速度比grep命令快，但是不够灵活：它只能找固定的文本，而不是规则表达式。如果你想在一个文件或者输出中找到包含星号字符的行：

		fgrep '*' /etc/profile
		
		for i in /etc/profile.d/*.sh ; do
		
		或
		
		grep -F '*' /etc/profile
		
		for i in /etc/profile.d/*.sh ; do


#五、source命令
功能：使Shell读入指定的Shell程序文件并依次执行文件中的所有语句
source命令通常用于重新执行刚修改的初始化文件，使之立即生效，而不必注销并重新登录。

用法：
source filename 或 . filename
source命令(从 C Shell 而来)是bash shell的内置命令;点命令(.)，就是个点符号(从Bourne Shell而来)是source的另一名称。

source filename 与 sh filename 及./filename执行脚本的区别在那里呢？

1、当shell脚本具有可执行权限时，用sh filename与./filename执行脚本是没有区别得。./filename是因为当前目录没有在PATH中，所有"."是用来表示当前目录的。

2、sh filename 重新建立一个子shell，在子shell中执行脚本里面的语句，该子shell继承父shell的环境变量，但子shell新建的、改变的变量不会被带回父shell，除非使用export。

3、source filename：这个命令其实只是简单地读取脚本里面的语句依次在当前shell里面执行，没有建立新的子shell。那么脚本里面所有新建、改变变量的语句都会保存在当前shell里面。


举例说明：
    1.新建一个test.sh脚本，内容为:A=1
    2.然后使其可执行chmod +x test.sh
    3.运行sh test.sh后，echo $A，显示为空，因为A=1并未传回给当前shell
    4.运行./test.sh后，也是一样的效果
    5.运行source test.sh 或者 . test.sh，然后echo $A，则会显示1，说明A=1的变量在当前shell中

#六、uname命令
>uname命令用于打印当前系统相关信息（内核版本号、硬件架构、主机名称和操作系统类型等）。

命令行选项：
> -a或--all：显示全部的信息；

> -m或--machine：显示电脑类型；

> -n或-nodename：显示在网络上的主机名称；

> -r或--release：显示操作系统的发行编号；

> -s或--sysname：显示操作系统名称；

> -v：显示操作系统的版本；

> -p或--processor：输出处理器类型或"unknown"； 
> 
> -i或--hardware-platform：输出硬件平台或"unknown"； 
> 
> -o或--operating-system：输出操作系统名称；
> 
>  --help：显示帮助；
>   
>  --version：显示版本信息。

#七、taskset命令
> taskset是Linux系统当中，用于查看、设定CPU核使用情况的命令
>语法：
>
>taskset [options] mask command [arg]...
>taskset [options] -p [mask] pid

taskset 命令用于设置或者获取一直指定的 PID 对于 CPU 核的运行依赖关系。也可以用 taskset 启动一个命令，直接设置它的 CPU 核的运行依赖关系。

CPU 核依赖关系是指，命令会被在指定的 CPU 核中运行，而不会再其他 CPU 核中运行的一种调度关系。需要说明的是，在正常情况下，为了系统性能的原因，调度器会尽可能的在一个 CPU 核中维持一个进程的执行。强制指定特殊的 CPU 核依赖关系对于特殊的应用是有意义的。

CPU 核的定义采用位定义的方式进行，最低位代表 CPU0，然后依次排序。这种位定义可以超过系统实际的 CPU 总数，并不会存在问题。通过命令获得的这种 CPU 位标记，只会包含系统实际 CPU 的数目。如果设定的位标记少于系统 CPU 的实际数目，那么命令会产生一个错误。当然这种给定的和获取的位标记采用 16 进制标识。

参数描述：

    -p, --pid
    对一个现有的进程进行操作，而不是启动一个新的进程
    -c, --cpu-list
    使用 CPU 编号替代位标记，这可以是一个列表，列表中可以使用逗号分隔，或者使用 "-" 进行范围标记，例如：0,5,7,9-11
    -h, --help
    打印帮助信息
    -V, --version
    打印版本信息
    
    使用默认的行为，用给定的 CPU 核运行位标记运行一个进程
    taskset mask command [arguments]
    获取一个指定进程的 CPU 核运行位标记
    taskset -p pid
    设定一个指定进程的 CPU 核运行位标记
    taskset -p mask pid