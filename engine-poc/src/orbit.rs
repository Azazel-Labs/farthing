//! An orbit camera driven by pointer and wheel input, for viewers that show one model.
//!
//! It turns around a target, pans across the view and dollies toward the target, easing out
//! after a drag as three.js's `OrbitControls` (with damping) does, so the web demo feels the
//! same as before it drew with this engine. It also keeps the target near the model and fits
//! the near and far planes around it.

use crate::viewer::Camera;
use farthing::glam::camera::rh::view::look_at_mat4;
use farthing::glam::{Mat4, Vec2, Vec3};

/// Part of a drag's motion applied each update; the rest carries over (damping).
const DAMPING: f32 = 0.05;
/// How much one wheel notch (100 units of `deltaY`) dollies: to 0.95 of the distance.
const ZOOM_BASE: f32 = 0.95;
/// Polar angles stay this far from straight up and down.
const POLE: f32 = 1e-6;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Gesture {
    Rotate,
    Pan,
    Dolly,
}

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub eye: Vec3,
    pub target: Vec3,
    /// Vertical field of view in degrees.
    pub fov_y: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    /// Whether dragging may move the target.
    pub pan_enabled: bool,
    /// The model's bounding sphere: the target stays within 1.5 radii of its center, and the
    /// near and far planes fit around it.
    pub focus: (Vec3, f32),
    /// The view's height in CSS pixels, which drags are measured against.
    pub height: f32,
    /// Rotation still to apply: azimuth and polar angle, radians.
    spin: Vec2,
    /// Target motion still to apply.
    pan_offset: Vec3,
    /// Distance factor for the next update.
    scale: f32,
    /// Pointers down: id and last position.
    pointers: Vec<(i32, Vec2)>,
    gesture: Gesture,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(2.0, 1.5, 3.0),
            target: Vec3::ZERO,
            fov_y: 60.0,
            min_distance: 0.0,
            max_distance: f32::INFINITY,
            pan_enabled: true,
            focus: (Vec3::ZERO, 1.0),
            height: 1.0,
            spin: Vec2::ZERO,
            pan_offset: Vec3::ZERO,
            scale: 1.0,
            pointers: Vec::new(),
            gesture: Gesture::Rotate,
        }
    }
}

impl OrbitCamera {
    /// The distance at which a sphere of `radius` fills the view, with a little room.
    pub fn home_distance(&self, radius: f32) -> f32 {
        radius / (self.fov_y.to_radians() / 2.0).sin() * 1.05
    }

    /// Looks at the focus sphere from the default angle, up and to the right.
    pub fn home(&mut self) {
        let (center, radius) = self.focus;
        self.target = center;
        self.eye = center + Vec3::new(0.6, 0.35, 1.0).normalize() * self.home_distance(radius);
        self.stop();
    }

    /// Drops any motion still easing out.
    pub fn stop(&mut self) {
        self.spin = Vec2::ZERO;
        self.pan_offset = Vec3::ZERO;
        self.scale = 1.0;
    }

    /// Moves the target to the focus center, the eye with it.
    pub fn recenter(&mut self) {
        let shift = self.focus.0 - self.target;
        self.target += shift;
        self.eye += shift;
        self.pan_offset = Vec3::ZERO;
    }

    /// Puts the eye `d` from the target, in the direction it is now.
    pub fn set_distance(&mut self, d: f32) {
        let dir = (self.eye - self.target).try_normalize().unwrap_or(Vec3::Z);
        self.eye = self.target + dir * d;
        self.scale = 1.0;
    }

    /// The eye's distance from the focus center.
    pub fn focus_distance(&self) -> f32 {
        self.eye.distance(self.focus.0)
    }

    pub fn pointer_down(&mut self, id: i32, at: Vec2, button: u32, modifier: bool) {
        self.pointers.retain(|p| p.0 != id);
        self.pointers.push((id, at));
        if self.pointers.len() == 1 {
            self.gesture = match button {
                1 => Gesture::Dolly,
                2 => Gesture::Pan,
                _ if modifier && self.pan_enabled => Gesture::Pan,
                _ => Gesture::Rotate,
            };
        }
    }

    pub fn pointer_move(&mut self, id: i32, at: Vec2) {
        let Some(i) = self.pointers.iter().position(|p| p.0 == id) else {
            return;
        };
        if self.pointers.len() >= 2 {
            // Two fingers: pinch to dolly, move together to pan.
            let other = self.pointers[1 - i.min(1)].1;
            let before = self.pointers[i].1;
            let (d0, d1) = (before.distance(other), at.distance(other));
            if d0 > 0.0 && d1 > 0.0 {
                self.scale *= d0 / d1;
            }
            if self.pan_enabled {
                self.pan((at - before) / 2.0);
            }
        } else {
            let delta = at - self.pointers[i].1;
            match self.gesture {
                Gesture::Rotate => {
                    self.spin -= std::f32::consts::TAU * delta / self.height.max(1.0)
                }
                Gesture::Pan if self.pan_enabled => self.pan(delta),
                Gesture::Pan => {}
                Gesture::Dolly => self.dolly(delta.y),
            }
        }
        self.pointers[i].1 = at;
    }

    pub fn pointer_up(&mut self, id: i32) {
        self.pointers.retain(|p| p.0 != id);
    }

    /// A wheel turn: `delta_y` as `WheelEvent.deltaY`, `mode` its `deltaMode` (0 pixels,
    /// 1 lines, 2 pages), `pinch` when the ctrl key is down (a trackpad pinch).
    pub fn wheel(&mut self, delta_y: f32, mode: u32, pinch: bool) {
        let unit = match mode {
            1 => 16.0,
            2 => 100.0,
            _ => 1.0,
        };
        self.dolly(delta_y * unit * if pinch { 10.0 } else { 1.0 });
    }

    /// Positive `amount` moves away, as scrolling down does.
    fn dolly(&mut self, amount: f32) {
        let s = ZOOM_BASE.powf((amount * 0.01).abs());
        if amount < 0.0 {
            self.scale *= s;
        } else if amount > 0.0 {
            self.scale /= s;
        }
    }

    /// Moves the target with the pointer: `delta` in CSS pixels, at the target's depth.
    fn pan(&mut self, delta: Vec2) {
        let offset = self.eye - self.target;
        let at_target = offset.length() * (self.fov_y.to_radians() / 2.0).tan();
        let (right, up) = self.axes();
        let per_px = 2.0 * at_target / self.height.max(1.0);
        self.pan_offset += -right * delta.x * per_px + up * delta.y * per_px;
    }

    /// The camera's right and up directions.
    fn axes(&self) -> (Vec3, Vec3) {
        let forward = (self.target - self.eye)
            .try_normalize()
            .unwrap_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        (right, right.cross(forward))
    }

    /// Applies part of the motion still to come, keeps the target near the model and fits the
    /// near and far planes. Call once per frame with the view's height in CSS pixels.
    pub fn update(&mut self, height: f32) {
        self.height = height;
        let offset = self.eye - self.target;
        let mut radius = offset.length();
        let mut theta = offset.x.atan2(offset.z);
        let mut phi = if radius > 0.0 {
            (offset.y / radius).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };
        theta += self.spin.x * DAMPING;
        phi = (phi + self.spin.y * DAMPING).clamp(POLE, std::f32::consts::PI - POLE);
        self.target += self.pan_offset * DAMPING;
        radius = (radius * self.scale).clamp(self.min_distance, self.max_distance);
        self.eye = self.target
            + radius * Vec3::new(phi.sin() * theta.sin(), phi.cos(), phi.sin() * theta.cos());
        self.spin *= 1.0 - DAMPING;
        self.pan_offset *= 1.0 - DAMPING;
        self.scale = 1.0;
        // Panning can't lose the model.
        let (center, r) = self.focus;
        let off = self.target - center;
        if off.length() > r * 1.5 {
            let shift = off.normalize() * r * 1.5 - off;
            self.target += shift;
            self.eye += shift;
        }
    }

    /// World to camera.
    pub fn view_matrix(&self) -> Mat4 {
        look_at_mat4(self.eye, self.target, Vec3::Y)
    }

    /// Near and far planes fit tightly around the focus sphere, for depth precision at any
    /// distance.
    pub fn planes(&self) -> (f32, f32) {
        let (center, r) = self.focus;
        let dist = self.eye.distance(center);
        ((r / 500.0).max((dist - r) * 0.5), dist + r * 2.0)
    }

    pub fn camera(&self) -> Camera {
        let (near, far) = self.planes();
        Camera {
            view: self.view_matrix(),
            fov_y: self.fov_y.to_radians(),
            near,
            far,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(c: &mut OrbitCamera) {
        for _ in 0..2000 {
            c.update(500.0);
        }
    }

    #[test]
    fn home_frames_the_focus() {
        let mut c = OrbitCamera {
            focus: (Vec3::new(1.0, 2.0, 3.0), 2.0),
            ..Default::default()
        };
        c.home();
        assert_eq!(c.target, c.focus.0);
        assert!((c.focus_distance() - c.home_distance(2.0)).abs() < 1e-4);
        let (near, far) = c.planes();
        assert!(near > 0.0 && near < c.focus_distance() - 2.0 && far > c.focus_distance() + 2.0);
    }

    #[test]
    fn a_drag_across_the_view_turns_all_the_way_round() {
        let mut c = OrbitCamera::default();
        c.home();
        let start = c.eye - c.target;
        c.pointer_down(1, Vec2::ZERO, 0, false);
        c.pointer_move(1, Vec2::new(500.0, 0.0));
        c.pointer_up(1);
        settle(&mut c);
        // A view height of drag is a full turn; damping eases it in over many frames.
        let end = c.eye - c.target;
        assert!(
            (end - start).length() < 1e-3 * start.length(),
            "{start} {end}"
        );
        assert!((end.length() - start.length()).abs() < 1e-4);
    }

    #[test]
    fn wheel_dollies_and_clamps() {
        let mut c = OrbitCamera {
            max_distance: 10.0,
            ..Default::default()
        };
        c.home();
        let d = c.focus_distance();
        c.wheel(-100.0, 0, false);
        c.update(500.0);
        assert!((c.focus_distance() - d * 0.95).abs() < 1e-4);
        for _ in 0..200 {
            c.wheel(100.0, 1, false);
            c.update(500.0);
        }
        assert!((c.focus_distance() - 10.0).abs() < 1e-3);
    }

    #[test]
    fn pan_moves_the_target_but_stays_near_the_model() {
        let mut c = OrbitCamera::default();
        c.home();
        c.pointer_down(1, Vec2::ZERO, 2, false);
        c.pointer_move(1, Vec2::new(-100.0, 0.0));
        c.pointer_up(1);
        settle(&mut c);
        let (right, _) = c.axes();
        assert!(c.target.dot(right) > 0.1);
        c.pointer_down(1, Vec2::ZERO, 2, false);
        c.pointer_move(1, Vec2::new(-1e5, 0.0));
        settle(&mut c);
        assert!(c.target.length() <= 1.5 + 1e-4);
        c.pan_enabled = false;
        let t = c.target;
        c.pointer_move(1, Vec2::new(-2e5, 0.0));
        settle(&mut c);
        assert_eq!(c.target, t);
    }

    #[test]
    fn pinch_dollies() {
        let mut c = OrbitCamera::default();
        c.home();
        let d = c.focus_distance();
        c.pointer_down(1, Vec2::new(0.0, 0.0), 0, false);
        c.pointer_down(2, Vec2::new(100.0, 0.0), 0, false);
        c.pointer_move(2, Vec2::new(200.0, 0.0));
        c.update(500.0);
        // The midpoint moved too, which pans: measure from the target.
        assert!(((c.eye - c.target).length() - d / 2.0).abs() < 1e-3);
    }
}
