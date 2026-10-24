# 旧文本进度管线归档

这里保存原 `qwen_training/` 顶层剩余的旧 checkpoint 评估、闭环 rollout、benchmark、
报告、测试和依赖文件，共 16 个文件。归档时保留文件内容。

这套管线使用文本生成输出和旧的六帧数据接口，不属于当前的 query token 方案。
其训练核心此前已被删除，本次没有恢复；因此本目录是历史参考，不是可直接运行的独立项目。
原始完整版本仍可从 Git 历史或现有 `backup/` 备份查找。

当前代码入口见项目根目录 README；RMBench 专用代码位于
`qwen_training/datasets/rmbench/`。
