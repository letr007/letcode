use serde::{Deserialize, Serialize};

use crate::permission::{PermissionMode, ToolScope};
use crate::tool_names;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCapabilityContract {
    pub name: String,
    pub purpose: String,
    pub tool_scope: ToolScope,
    pub permission_mode: PermissionMode,
    pub can_write: bool,
    pub can_delegate: bool,
    pub default_timeout_secs: Option<u64>,
    pub default_max_tool_calls: Option<usize>,
    pub input_expectations: String,
    pub expected_result_shape: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubagentCatalogEntry {
    pub agent_name: &'static str,
    pub tool_name: &'static str,
    pub task_description: &'static str,
    pub tool_description: &'static str,
    pub read_only: bool,
}

pub(crate) const SUBAGENT_CATALOG: &[SubagentCatalogEntry] = &[
    SubagentCatalogEntry {
        agent_name: "explorer",
        tool_name: tool_names::TOOL_AGENT_EXPLORE,
        task_description: "交给 explorer 子代理执行的聚焦只读调研任务",
        tool_description: "将限定范围的只读仓库调研任务委派给 explorer 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "fixer",
        tool_name: tool_names::TOOL_AGENT_FIXER,
        task_description: "交给 fixer 子代理执行的聚焦实现或修复任务",
        tool_description: "将限定范围的实现或修复任务委派给 fixer 子代理，并返回摘要。",
        read_only: false,
    },
    SubagentCatalogEntry {
        agent_name: "oracle",
        tool_name: tool_names::TOOL_AGENT_ORACLE,
        task_description: "交给 oracle 子代理执行的根因分析、风险判断或验证建议任务",
        tool_description: "将限定范围的根因分析、风险判断或验证建议任务委派给 oracle 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "designer",
        tool_name: tool_names::TOOL_AGENT_DESIGNER,
        task_description: "交给 designer 子代理执行的设计、方案整理或接口梳理任务",
        tool_description: "将限定范围的设计、方案整理或接口梳理任务委派给 designer 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "librarian",
        tool_name: tool_names::TOOL_AGENT_LIBRARIAN,
        task_description: "交给 librarian 子代理执行的资料整理、证据检索或上下文归档任务",
        tool_description: "将限定范围的仓库资料整理、证据检索或上下文归档任务委派给 librarian 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "general",
        tool_name: tool_names::TOOL_AGENT_GENERAL,
        task_description: "交给 general 子代理执行的限定范围只读通用辅助任务",
        tool_description: "将限定范围的只读通用辅助任务委派给 general 子代理，并返回摘要。",
        read_only: true,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTemplate {
    pub name: String,
    pub purpose: String,
    pub system_prompt: String,
    pub tool_scope: ToolScope,
    pub permission_mode: PermissionMode,
    pub can_write: bool,
    pub can_delegate: bool,
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub input_expectations: String,
    pub expected_result_shape: String,
}

impl AgentTemplate {
    fn read_only(name: &str, purpose: &str, system_prompt: &str) -> Self {
        Self {
            name: name.into(),
            purpose: purpose.into(),
            system_prompt: system_prompt.into(),
            tool_scope: ToolScope::ReadOnlyExplorer,
            permission_mode: PermissionMode::Default,
            can_write: false,
            can_delegate: false,
            timeout_secs: None,
            max_tool_calls: None,
            input_expectations: "需要明确的 task 或 objective；可选 success_criteria、allowed_paths、forbidden_paths、owned_paths。runtime 超时和工具预算由配置继承，不应在普通委派里填写。".into(),
            expected_result_shape: "JSON object with run_id, child_session_id, agent_name, status, summary.".into(),
        }
    }

    pub fn explorer() -> Self {
        Self::read_only(
            "explorer",
            "只读仓库探索",
            concat!(
                "你是一个只读的 explorer 子代理。请围绕分配给你的任务调查本地项目，仓库，文件夹等、给出结论，",
                "并且只能使用只读工具。不要编辑文件，不要运行具备写能力的命令，也不要继续委派。"
            ),
        )
    }

    pub fn fixer() -> Self {
        Self {
            name: "fixer".into(),
            purpose: "修复/构建者代理".into(),
            system_prompt: concat!("你是一个可读可写的修复者子代理。根据主代理给出的方向和要求，使用合理的工具，按照意图进行实现。", "请严格按照主代理的要求来进行实现，而非自己想当然的做法。仅做主代理要求做的部分，不做分外的事。", "你可以使用绝大多数工具，但请按照要求来。").into(),
            tool_scope: ToolScope::FullAccess,
            permission_mode: PermissionMode::Default,
            can_write: true,
            can_delegate: false,
            timeout_secs: None,
            max_tool_calls: None,
            input_expectations: "需要明确的 task 或 objective；可选 success_criteria、allowed_paths、forbidden_paths、owned_paths。runtime 超时和工具预算由配置继承，不应在普通委派里填写。".into(),
            expected_result_shape: "JSON object with run_id, child_session_id, agent_name, status, summary.".into(),
        }
    }

    pub fn oracle() -> Self {
        Self::read_only(
            "oracle",
            "只读根因与风险分析",
            concat!(
                "你是 oracle 子代理。专注于只读分析、根因判断、方案权衡、风险识别与验证建议。",
                "不要修改文件，不要运行具备写能力的命令，不要继续委派。输出应帮助主代理做决策，而不是代替 fixer 实现修改。"
            ),
        )
    }

    pub fn designer() -> Self {
        Self::read_only(
            "designer",
            "只读设计与方案整理",
            concat!(
                "你是 designer 子代理。专注于阅读现有实现、梳理接口、提出小而清晰的设计方案、命名建议与变更边界。",
                "不要修改文件，不要运行具备写能力的命令，不要继续委派。"
            ),
        )
    }

    pub fn librarian() -> Self {
        Self::read_only(
            "librarian",
            "只读资料整理与证据归档",
            concat!(
                "你是 librarian 子代理。专注于检索本仓库中的相关文件、证据、历史上下文、接口位置与约束，",
                "给出紧凑且可追溯的引用。不要修改文件，不要运行具备写能力的命令，不要继续委派。"
            ),
        )
    }

    pub fn general() -> Self {
        Self::read_only(
            "general",
            "只读通用问题助手",
            concat!(
                "你是 general 子代理。用于边界明确但不属于其他专家的只读辅助任务，例如梳理奇怪输出、归纳现象、总结仓库事实。",
                "保持只读，不要实现修改，不要替代 fixer，不要继续委派。"
            ),
        )
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "explorer" => Some(Self::explorer()),
            "fixer" => Some(Self::fixer()),
            "oracle" => Some(Self::oracle()),
            "designer" => Some(Self::designer()),
            "librarian" => Some(Self::librarian()),
            "general" => Some(Self::general()),
            _ => None,
        }
    }

    pub fn catalog() -> Vec<Self> {
        vec![
            Self::explorer(),
            Self::fixer(),
            Self::oracle(),
            Self::designer(),
            Self::librarian(),
            Self::general(),
        ]
    }

    pub fn capability_contract(&self) -> SubagentCapabilityContract {
        SubagentCapabilityContract {
            name: self.name.clone(),
            purpose: self.purpose.clone(),
            tool_scope: self.tool_scope,
            permission_mode: self.permission_mode,
            can_write: self.can_write,
            can_delegate: self.can_delegate,
            default_timeout_secs: self.timeout_secs,
            default_max_tool_calls: self.max_tool_calls,
            input_expectations: self.input_expectations.clone(),
            expected_result_shape: self.expected_result_shape.clone(),
        }
    }
}

pub struct AgentFactory;

pub(crate) fn is_subagent_tool_name(name: &str) -> bool {
    agent_name_for_subagent_tool(name).is_some()
}

pub(crate) fn agent_name_for_subagent_tool(tool_name: &str) -> Option<&'static str> {
    subagent_catalog_entry_by_tool_name(tool_name).map(|entry| entry.agent_name)
}

pub(crate) fn subagent_tool_name_for_agent_name(agent_name: &str) -> Option<&'static str> {
    subagent_catalog_entry_by_agent_name(agent_name).map(|entry| entry.tool_name)
}

pub(crate) fn subagent_catalog_entry_by_tool_name(
    tool_name: &str,
) -> Option<&'static SubagentCatalogEntry> {
    SUBAGENT_CATALOG
        .iter()
        .find(|entry| entry.tool_name == tool_name)
}

pub(crate) fn subagent_catalog_entry_by_agent_name(
    agent_name: &str,
) -> Option<&'static SubagentCatalogEntry> {
    SUBAGENT_CATALOG
        .iter()
        .find(|entry| entry.agent_name == agent_name)
}
