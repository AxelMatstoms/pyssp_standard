from pyssp_standard.standard import ModelicaStandard
from pyssp_standard.common_content_ssc import BaseElement
from pyssp_standard.traceability import TaskMetaData, GPhaseCommon, Step, make_phase


@make_phase
class AnalysisPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    analyze_simulation_task_and_objectives: Step
    verify_analysis: Step


@make_phase
class RequirementsPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    define_model_requirements: Step
    define_parameter_requirements: Step
    define_simulation_environment_requirements: Step
    define_simulation_integration_requirements: Step
    define_test_case_requirements: Step
    define_quality_assurance_requirements: Step
    verify_requirements: Step


@make_phase
class DesignPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    define_model_design_specification: Step
    define_parameter_design_specification: Step
    define_simulation_environment_design_specification: Step
    define_simulation_integration_design_specification: Step
    define_test_case_design_specification: Step
    define_quality_assurance_design_specification: Step
    verify_design_specification: Step


@make_phase
class ImplementationPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    implement_model: Step
    implement_parameter: Step
    implement_simulation_environment: Step
    implement_test_case: Step
    integrate_simulation: Step
    assure_simulation_setup_quality: Step
    derive_simulation_setup_quality_verdict: Step


@make_phase
class ExecutionPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    execute_simulation: Step


@make_phase
class EvaluationPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    evaluate_simulation_results: Step
    assure_simulation_quality: Step
    derive_simulation_quality_verdict: Step


@make_phase
class FulfillmentPhase(GPhaseCommon, BaseElement, ModelicaStandard):
    decide_simulation_objective_fulfillment: Step


class STMD(TaskMetaData):
    analysis_phase: AnalysisPhase
    requirements_phase: RequirementsPhase
    design_phase: DesignPhase
    implementation_phase: ImplementationPhase
    execution_phase: ExecutionPhase
    evaluation_phase: EvaluationPhase
    fulfillment_phase: FulfillmentPhase

    ns = "stmd"
    tag_name = "SimulationTaskMetaData"

    def __init__(self, file_path, mode="r"):
        self.analysis_phase = None
        self.requirements_phase = None
        self.design_phase = None
        self.implementation_phase = None
        self.execution_phase = None
        self.evaluation_phase = None
        self.fulfillment_phase = None

        super().__init__(file_path, mode, "stmd11")

    def __read__(self):
        super().__read__()

        fields = STMD.__annotations__
        for name, cls in fields.items():
            if (elem := self.root.find(f"stmd:{cls.__name__}", self.namespaces)) is not None:
                setattr(self, name, cls(elem, self.resource_manager))

    def __write__(self):
        super().__write__()

        fields = STMD.__annotations__
        for name, cls in fields.items():
            if (phase := getattr(self, name)) is not None:
                self.root.append(phase.as_element())

        self.common.update_element(self.root)
