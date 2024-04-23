//! The element context is the main interface for interacting with the frame during a paint.
//!
//! Elements are hierarchical and with a few exceptions the context accumulates state in a stack
//! as it processes all of the elements in the frame. The methods that interact with this stack
//! are generally marked with `with_*`, and take a callback to denote the region of code that
//! should be executed with that state.
//!
//! The other main interface is the `paint_*` family of methods, which push basic drawing commands
//! to the GPU. Everything in a GPUI app is drawn with these methods.
//!
//! There are also several internal methods that GPUI uses, such as [`ElementContext::with_element_state`]
//! to call the paint and layout methods on elements. These have been included as they're often useful
//! for taking manual control of the layouting or painting of specialized elements.

use std::{
    any::{Any, TypeId},
    borrow::{Borrow, BorrowMut, Cow},
    cmp, mem,
    ops::Range,
    rc::Rc,
    sync::Arc,
};

use anyhow::Result;
use collections::FxHashMap;
use derive_more::{Deref, DerefMut};
use futures::{future::Shared, FutureExt};
#[cfg(target_os = "macos")]
use media::core_video::CVImageBuffer;
use smallvec::SmallVec;
use util::post_inc;

use crate::{
    hash, point, prelude::*, px, size, AnyElement, AnyTooltip, AppContext, Asset, AvailableSpace,
    Bounds, BoxShadow, ContentMask, Corners, CursorStyle, DevicePixels, DispatchNodeId,
    DispatchPhase, DispatchTree, DrawPhase, ElementId, ElementStateBox, EntityId, FocusHandle,
    FocusId, FontId, GlobalElementId, GlyphId, Hsla, ImageData, InputHandler, IsZero, KeyContext,
    KeyEvent, LayoutId, LineLayoutIndex, ModifiersChangedEvent, MonochromeSprite, MouseEvent,
    PaintQuad, Path, Pixels, PlatformInputHandler, Point, PolychromeSprite, Quad,
    RenderGlyphParams, RenderImageParams, RenderSvgParams, Scene, Shadow, SharedString, Size,
    StrikethroughStyle, Style, Task, TextStyleRefinement, TransformationMatrix, Underline,
    UnderlineStyle, Window, WindowContext, SUBPIXEL_VARIANTS,
};

/// This context is used for assisting in the implementation of the element trait
#[derive(Deref, DerefMut)]
pub struct ElementContext<'a> {
    pub(crate) cx: WindowContext<'a>,
}

impl<'a> WindowContext<'a> {
    /// Convert this window context into an ElementContext in this callback.
    /// If you need to use this method, you're probably intermixing the imperative
    /// and declarative APIs, which is not recommended.
    pub fn with_element_context<R>(&mut self, f: impl FnOnce(&mut ElementContext) -> R) -> R {
        f(&mut ElementContext {
            cx: WindowContext::new(self.app, self.window),
        })
    }
}

impl<'a> Borrow<AppContext> for ElementContext<'a> {
    fn borrow(&self) -> &AppContext {
        self.cx.app
    }
}

impl<'a> BorrowMut<AppContext> for ElementContext<'a> {
    fn borrow_mut(&mut self) -> &mut AppContext {
        self.cx.borrow_mut()
    }
}

impl<'a> Borrow<WindowContext<'a>> for ElementContext<'a> {
    fn borrow(&self) -> &WindowContext<'a> {
        &self.cx
    }
}

impl<'a> BorrowMut<WindowContext<'a>> for ElementContext<'a> {
    fn borrow_mut(&mut self) -> &mut WindowContext<'a> {
        &mut self.cx
    }
}

impl<'a> Borrow<Window> for ElementContext<'a> {
    fn borrow(&self) -> &Window {
        self.cx.window
    }
}

impl<'a> BorrowMut<Window> for ElementContext<'a> {
    fn borrow_mut(&mut self) -> &mut Window {
        self.cx.borrow_mut()
    }
}

impl<'a> Context for ElementContext<'a> {
    type Result<T> = <WindowContext<'a> as Context>::Result<T>;

    fn new_model<T: 'static>(
        &mut self,
        build_model: impl FnOnce(&mut crate::ModelContext<'_, T>) -> T,
    ) -> Self::Result<crate::Model<T>> {
        self.cx.new_model(build_model)
    }

    fn reserve_model<T: 'static>(&mut self) -> Self::Result<crate::Reservation<T>> {
        self.cx.reserve_model()
    }

    fn insert_model<T: 'static>(
        &mut self,
        reservation: crate::Reservation<T>,
        build_model: impl FnOnce(&mut crate::ModelContext<'_, T>) -> T,
    ) -> Self::Result<crate::Model<T>> {
        self.cx.insert_model(reservation, build_model)
    }

    fn update_model<T, R>(
        &mut self,
        handle: &crate::Model<T>,
        update: impl FnOnce(&mut T, &mut crate::ModelContext<'_, T>) -> R,
    ) -> Self::Result<R>
    where
        T: 'static,
    {
        self.cx.update_model(handle, update)
    }

    fn read_model<T, R>(
        &self,
        handle: &crate::Model<T>,
        read: impl FnOnce(&T, &AppContext) -> R,
    ) -> Self::Result<R>
    where
        T: 'static,
    {
        self.cx.read_model(handle, read)
    }

    fn update_window<T, F>(&mut self, window: crate::AnyWindowHandle, f: F) -> Result<T>
    where
        F: FnOnce(crate::AnyView, &mut WindowContext<'_>) -> T,
    {
        self.cx.update_window(window, f)
    }

    fn read_window<T, R>(
        &self,
        window: &crate::WindowHandle<T>,
        read: impl FnOnce(crate::View<T>, &AppContext) -> R,
    ) -> Result<R>
    where
        T: 'static,
    {
        self.cx.read_window(window, read)
    }
}

impl<'a> VisualContext for ElementContext<'a> {
    fn new_view<V>(
        &mut self,
        build_view: impl FnOnce(&mut crate::ViewContext<'_, V>) -> V,
    ) -> Self::Result<crate::View<V>>
    where
        V: 'static + Render,
    {
        self.cx.new_view(build_view)
    }

    fn update_view<V: 'static, R>(
        &mut self,
        view: &crate::View<V>,
        update: impl FnOnce(&mut V, &mut crate::ViewContext<'_, V>) -> R,
    ) -> Self::Result<R> {
        self.cx.update_view(view, update)
    }

    fn replace_root_view<V>(
        &mut self,
        build_view: impl FnOnce(&mut crate::ViewContext<'_, V>) -> V,
    ) -> Self::Result<crate::View<V>>
    where
        V: 'static + Render,
    {
        self.cx.replace_root_view(build_view)
    }

    fn focus_view<V>(&mut self, view: &crate::View<V>) -> Self::Result<()>
    where
        V: crate::FocusableView,
    {
        self.cx.focus_view(view)
    }

    fn dismiss_view<V>(&mut self, view: &crate::View<V>) -> Self::Result<()>
    where
        V: crate::ManagedView,
    {
        self.cx.dismiss_view(view)
    }
}

impl<'a> ElementContext<'a> {}
