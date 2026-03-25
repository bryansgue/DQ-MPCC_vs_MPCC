"""sim_control.py — Utilidad aislada para controlar el simulador MuJoCo desde ROS2.

Uso como módulo en tu controlador:
    from sim_control import SimControl
    sc = SimControl(node)
    sc.reset()              # reset síncrono (espera respuesta)
    sc.reset_async()        # reset no bloqueante (fire-and-forget)

Uso standalone:
    python3 sim_control.py              # reset una vez y sale
    python3 sim_control.py --wait       # espera confirmación interactiva
"""

import sys
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


RESET_SERVICE = "sim/reset"   # relativo al namespace del drone
DRONE_NS      = "/quadrotor"  # namespace del drone (debe coincidir con quad_name)


class SimControl:
    """Wrapper para llamar servicios de control del simulador.

    Puede usarse integrado en un nodo existente (pasando el node)
    o de forma autónoma (crea su propio nodo interno).
    """

    def __init__(self, node: Node = None, drone_ns: str = DRONE_NS):
        self._own_node = node is None
        if self._own_node:
            self._node = rclpy.create_node("sim_control")
        else:
            self._node = node

        service_name = f"{drone_ns}/{RESET_SERVICE}"
        self._reset_cli = self._node.create_client(Trigger, service_name)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def reset(self, timeout_sec: float = 3.0) -> bool:
        """Reset síncrono: bloquea hasta recibir respuesta del simulador.

        Returns True si el reset fue exitoso, False si hubo timeout.
        """
        if not self._reset_cli.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().warn(
                f"[SimControl] Servicio {self._reset_cli.srv_name} no disponible"
            )
            return False

        future = self._reset_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_sec)

        if future.done():
            resp = future.result()
            self._node.get_logger().info(
                f"[SimControl] Reset: success={resp.success}  msg='{resp.message}'"
            )
            return resp.success

        self._node.get_logger().warn("[SimControl] Reset timeout")
        return False

    def reset_async(self) -> None:
        """Reset no bloqueante: envía la petición y continúa sin esperar."""
        if not self._reset_cli.service_is_ready():
            self._node.get_logger().warn("[SimControl] reset_async: servicio no listo")
            return
        self._reset_cli.call_async(Trigger.Request())

    def destroy(self) -> None:
        """Liberar recursos si SimControl creó su propio nodo."""
        if self._own_node:
            self._node.destroy_node()


# ------------------------------------------------------------------
# Uso standalone
# ------------------------------------------------------------------

def main():
    rclpy.init()
    sc = SimControl()

    print(f"[sim_control] Solicitando reset en {DRONE_NS}/{RESET_SERVICE} ...")
    ok = sc.reset()
    print(f"[sim_control] {'OK' if ok else 'FALLO'}")

    sc.destroy()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
