#Python 字符串前缀r、u、b、f含义
##1、r/R表示raw string（原始字符串）
>在普通字符串中，反斜线是转义符，代表一些特殊的内容，如换行符\n，前缀r表示该字符串是原始字符串，即\不是转义符，只是单纯的一个符号。常用于特殊的字符如换行符、正则表达式、文件路径。
>
>代码示例：
>
	  str1 = "Hello\nWorld"
	  str2 = r"Hello\nWorld"
	  print(str1)
	  print(str2)
      #打印结果：
      Hello
      World
      Hello \n World
##2、u/U表示unicode string（unicode编码字符串）
>前缀u表示该字符串是unicode编码，Python2中用，用在含有中文字符的字符串前，防止因为编码问题，导致中文出现乱码。另外一般要在文件开关标明编码方式采用utf8。Python3中，所有字符串默认都是unicode字符串。 一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码
>   
    str1 = '\u4f60\u597d\u4e16\u754c'
	str2 = u'\u4f60\u597d\u4e16\u754c'
	print(str1)
	print(str2)
	# 打印结果如下：
	你好世界
	你好世界

##3、b/B表示byte string（转换成bytes类型）
>常用在如网络编程中，服务器和浏览器只认bytes类型数据。如：send 函数的参数和 recv 函数的返回值都是 bytes 类型。
>   
    str1 = 'hello world'
	str2 = b'hello world'
	print(type(str1))
	print(type(str2))
	# 打印结果如下：
	<class 'str'>
	<class 'bytes'>

##4、f/F表示format string（格式化字符串）
>前缀f用来格式化字符串。可以看出f前缀可以更方便的格式化字符串,比format()方法可读性高且使用方便。
>
	name = "张三"
	age = 20
	print(f"我叫{name},今年{age}岁。")
	# 打印结果如下：
	我叫张三,今年20岁。
    #如果把 f 删掉，打印结果如下：
    我叫{name},今年{age}岁。

>    
    $(pwd)：表示当前所在目录。
	
	${ }：获取变量的结果。一般情况下，$var与${var}是没有区别的，但是用${ }会比较精确的界定变量名称的范围。
	
	$0：shell 命令本身。
	
	$1 到 $9： 表示 Shell 的第几个参数。
	
	$? ：显示最后命令的执行情况
	
	$#：传递到脚本的参数个数
	
	$$：脚本运行的当前进程 ID 号
	
	$*：以一个单字符串显示所有向脚本传递的参数
	
	$!：后台运行的最后一个进程的 ID 号
	
	$-：显示 Shell 使用的当前选项