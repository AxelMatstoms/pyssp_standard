from pyssp_standard.standard import ModelicaStandard
from pyssp_standard.common_content_ssc import BaseElement
from pyssp_standard.traceability import TaskMetaData, GPhaseCommon, Step, make_phase


@make_phase
class AnalysisPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    analyze_decision_task: Step
    verify_analysis: Step


@make_phase
class DefinitionPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    define_sub_tasks: Step
    define_result_quality: Step
    verify_sub_tasks: Step


@make_phase
class ExecutionPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    execute_sub_tasks: Step


@make_phase
class EvaluationPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    evaluate_results: Step
    assure_result_quality: Step
    derive_result_quality_verdict: Step


@make_phase
class FulfillmentPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    decide_objective_fulfillment: Step


class STMD(TaskMetaData):
    analysis_phase: AnalysisPhase
    definition_phase: DefinitionPhase
    execution_phase: ExecutionPhase
    evaluation_phase: EvaluationPhase
    fulfillment_phase: FulfillmentPhase

    ns = "dtmd"
    tag_name = "DecisionTaskMetaData"

    def __init__(self, file_path, mode="r"):
        self.analysis_phase = None
        self.definition_phase = None
        self.execution_phase = None
        self.evaluation_phase = None
        self.fulfillment_phase = None

        super().__init__(file_path, mode, "dtmd11")

    def __read__(self):
        super().__read__()

        fields = STMD.__annotations__
        for name, cls in fields.items():
            if (elem := self.root.find(f"dtmd:{cls.__name__}", self.namespaces)) is not None:
                setattr(self, name, cls(elem, self.resource_manager))

    def __write__(self):
        super().__write__()

        fields = STMD.__annotations__
        for name, cls in fields.items():
            if (phase := getattr(self, name)) is not None:
                self.root.append(phase.as_element())

        self.common.update_element(self.root)
