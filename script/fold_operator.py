import onnx
from onnx import helper


def replace_with_softmax(onnx_model_path, output_model_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    nodes_to_remove = []
    target_input = None
    target_output = None

    for node in graph.node:
        # Check Exp -> ReduceSum -> Div Mode
        if node.op_type == "Exp":
            exp_output = node.output[0]
            for next_node in graph.node:
                if next_node.op_type == "ReduceSum" and exp_output in next_node.input:
                    reduce_output = next_node.output[0]
                    for final_node in graph.node:
                        if final_node.op_type == "Div" and reduce_output in final_node.input:
                            nodes_to_remove.extend([node, next_node, final_node])
                            target_input = node.input[0]
                            target_output = final_node.output[0]
                            break

    if not target_input or not target_output:
        print("Can not find Exp -> ReduceSum -> Div mode.")
        return

    for node in nodes_to_remove:
        graph.node.remove(node)

    softmax_node = helper.make_node(
        "Softmax",
        inputs=[target_input],
        outputs=[target_output],
        axis=1
    )
    graph.node.append(softmax_node)

    onnx.save(model, output_model_path)
    print(f"Model has been updated and saved in {output_model_path}")


input_path = "../model/model.onnx.old"
output_path = "../model/model.onnx.new"

replace_with_softmax(input_path, output_path)
