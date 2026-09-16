"""
gui/node_editor/evaluator.py
Zunifikowany, modularny silnik obliczeniowy grafu DAG.
Odpowiada za sortowanie topologiczne (algorytm Kahna) oraz propagację
danych numerycznych i nazw pomiędzy węzłami potoku.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger("SpectroHub")


class DAGEvaluator:
    """Modularny silnik ewaluacji potoku węzłowego."""

    @staticmethod
    def topological_sort(nodes: list) -> Tuple[Optional[list], bool]:
        """Sortowanie topologiczne acyklicznego grafu skierowanego (Kahn)."""
        adj = {n: [] for n in nodes}
        in_degree = {n: 0 for n in nodes}

        for n in nodes:
            for in_p in n.in_ports.values():
                for conn in in_p.connections:
                    if conn.out_port and conn.out_port.node_item:
                        src = conn.out_port.node_item
                        if src in adj:
                            adj[src].append(n)
                            in_degree[n] += 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(nodes):
            return None, True
        return ordered, False

    @staticmethod
    def evaluate_dag(
        ordered_nodes: list,
        default_x: np.ndarray,
        default_y: np.ndarray,
        session_spectra: Any = None,
        is_execution: bool = False,
        progress_callback: Optional[Callable[[int, int, float], bool]] = None
    ) -> Tuple[
        Dict[Tuple[Any, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        Dict[Tuple[Any, str], List[str]]
    ]:
        """Ewaluuje acykliczny potok DAG, delegując obliczenia do obiektów BaseNode."""
        port_results: Dict[Tuple[Any, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        port_names: Dict[Tuple[Any, str], List[str]] = {}

        for node in ordered_nodes:
            if not getattr(node, 'node_obj', None):
                continue

            node_inputs: Dict[str, Any] = {}
            node_input_names: Dict[str, Any] = {}

            # Mapowanie połączeń z portów wejściowych węzła
            for port_id, in_port in node.in_ports.items():
                if in_port.connections:
                    if in_port.allow_multiple:
                        multi_data = []
                        multi_names = []
                        for conn in in_port.connections:
                            if conn.out_port and conn.out_port.node_item:
                                k = (conn.out_port.node_item, conn.out_port.port_id)
                                ds = port_results.get(k, (None, None))
                                ns = port_names.get(k, [])
                                if ds[0] is not None:
                                    multi_data.append(ds)
                                    multi_names.append(ns)
                        node_inputs[f"{port_id}_multi"] = multi_data
                        node_input_names[f"{port_id}_multi"] = multi_names
                        node_inputs[port_id] = multi_data[0] if multi_data else (None, None)
                        node_input_names[port_id] = multi_names[0] if multi_names else []
                    else:
                        conn = in_port.connections[0]
                        if conn.out_port and conn.out_port.node_item:
                            src_item = conn.out_port.node_item
                            src_p_id = conn.out_port.port_id
                            k = (src_item, src_p_id)
                            node_inputs[port_id] = port_results.get(k, (None, None))
                            node_input_names[port_id] = port_names.get(k, [])

                            # --- NOWOŚĆ: automatyczne przekazanie paczki danych HDO ---
                            src_obj = getattr(src_item, 'node_obj', src_item)
                            bundles = getattr(src_obj, 'runtime_bundles', {})
                            if src_p_id in bundles:
                                node_inputs[f"{port_id}_bundle"] = bundles[src_p_id]
                else:
                    node_inputs[port_id] = (None, None)
                    node_input_names[port_id] = []

            # Dodatkowy kontekst dla Solwera HDO
            if node.node_type == "hdo_decomposition":
                p_bas = node.in_ports.get("in_bases")
                b_list = []
                if p_bas:
                    for conn in p_bas.connections:
                        if conn.out_port and conn.out_port.node_item:
                            sn = conn.out_port.node_item
                            w_min = float(sn.step_data.get('w_min', 0.0))
                            w_max = float(sn.step_data.get('w_max', 100.0))
                            b_list.append((w_min, w_max))
                node_inputs["in_bases_bounds"] = b_list

                p_pks = node.in_ports.get("in_peaks")
                pk_params = []
                if p_pks:
                    for conn in p_pks.connections:
                        if conn.out_port and conn.out_port.node_item:
                            pk_params.append(conn.out_port.node_item.step_data)
                node_inputs["in_peaks_params"] = pk_params

                # Pobieramy konfigurację z portu in_cfg
                cfg_node = None
                p_cfg = node.in_ports.get("in_cfg")
                if p_cfg and p_cfg.connections and p_cfg.connections[0].out_port:
                    cfg_node = p_cfg.connections[0].out_port.node_item
                else:
                    for other in ordered_nodes:
                        if other.node_type == "hdo_config":
                            cfg_node = other
                            break
                node_inputs["in_cfg_config"] = cfg_node.step_data if cfg_node else {}

            # Wywołanie czystej metody compute() w węźle
            try:
                out_results, out_names = node.node_obj.compute(
                    inputs=node_inputs,
                    input_names=node_input_names,
                    is_execution=is_execution,
                    progress_callback=progress_callback
                )
                for p_id, data_tuple in out_results.items():
                    port_results[(node, p_id)] = data_tuple
                for p_id, names_list in out_names.items():
                    port_names[(node, p_id)] = names_list
            except Exception as e:
                logger.warning(f"Błąd kalkulacji węzła {node.title}: {e}")

        return port_results, port_names