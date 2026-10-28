# Architecture

## Modules

- `app.py`: `App`, the window. The rail lists `LESSONS`; under the current stage, its
  `sections` (a click sets `lesson.jump`, which scrolls the page there); below them the stage's `controls()`. The
  page on the right is `lesson.page()`. The menu bar has Stages and, when imgui_debugger is installed (the `debug`
  extra), its `DebugTools` (inspector targeted at the current page, style editor, imgui's tools). `run(argv)` is
  the `imgui-ophys` entry point.
- `lesson.py`: `Lesson`, the base class: `key`, `title`, `summary`, `sections`, `group` (empty for a pipeline
  stage; otherwise the rail lists it unnumbered under that header, after the stages), `references` ((citation, url),
  drawn as a last References subsection); `page()` draws `section_<i>()` in order and honours `jump`. `Lesson.goto(key)` (set by `App`) opens another stage.
- `lessons/<name>/model.py`: one stage's numbers, pure numpy, tested without imgui.
- `lessons/<name>/view.py`: the stage's `Lesson` subclass.
- `sim.py`: Poisson spikes, calcium kernels and a membrane model with GEVI kinetics on a fine grid, averaged into
  camera frames, then photons per frame with shot, read, neuropil and bleaching kept as separate parts
  (`Trace.parts`, `Trace.on`). Baselines (F0), denoisers and the isolated-event average.
- `diagram.py`: draw-list figures: `matrix` (a 2-D array cell by cell, values written in, rows / columns / cells
  outlined, returns the hovered cell), `stack` (frames as a deck), `sparkline`, `ticks`, `box`, `arrow`,
  `chip_row`, `segments` (colored text runs), `heading`, `canvas`.
- `plotting.py`: colors, `spec`, `begin_plot` (legend above the plot), `Colormaps`, `image_plot` (heatmap, row 0
  on top, click or drag to pick a pixel), `trace_plot` (lines with a time cursor).
- `widgets.py`: `section`, `help_mark`, `operator`, `table` (key / meaning rows), `push_size`, `u32`.
- `config.py`: `~/.imgui_ophys/settings.json` and the UI scale.

The only hard dependencies are imgui-bundle and numpy. imgui_debugger is optional, imported in a try in `app.py`. The
scale handling and plotting helpers follow imgui_vnoiser.

## Frame

`App.gui` applies the scale, calls `lesson.update(dt)`, then draws. While `lesson.animating()` is true, hello_imgui's
idling is off so playback and iterative fits run at full rate; otherwise the window idles at 20 fps.

## Pages

- Motion correction (`motion/model.py`): a field of cells moved by a random walk plus a smooth bend (shifts set at a
  3 x 3 grid and interpolated per pixel), sampled between pixels. Rigid: one shift per frame; piecewise rigid: a
  shift per 16 x 16 patch (half overlapping), interpolated into a per-pixel shift, frame warped back, so no seams.
  Shifts: every whole-pixel candidate scored against the moved reference in one matrix product, then a parabola
  through the peak for the part between pixels. The reference follows suite2p (`pick_initial_reference`,
  `compute_reference`).
- Demixing (NMF): `guide.py` follows `make_tiny()` (6 x 6 x 24) from frames to footprints; HALS steps `HalsWalk`, one
  trace or footprint update at a time (a sweep equals `NMF.step`, tested). NMF vs PCA fits `make_movie()`
  (24 x 24 x 300) live.
- Traces: `sim.make_trace` once per parameter change; the parts switch on and off without resimulating
  (`Trace.on`). dF/F uses F0 = the 20th percentile of what was recorded, over the whole recording or a running
  window.
- Signal quality (`dprime/model.py`): d' as in the jGCaMP8 paper, (mean top - mean bottom) / pooled SD, on the
  Traces simulator with the spike times known: `score` takes a window of frames before each isolated spike and one
  from the response peak on, pooled across spikes; `accuracy` is Phi(d' / 2). `noise_rows` adds white noise to a
  clean trace by bisection until d' lands on each target, for the example rows. The view follows one spike
  (median amplitude) and keeps its plot ranges fixed over the reference cells, so a light or bleaching change moves
  the dots instead of the axes.
- SVD (under Other, `svd/model.py`): `Peel` finds the pieces one at a time by power iteration and deflation (u = R v,
  v = R' u until v settles, then subtract s u v'), one half-step per click; its pieces equal `np.linalg.svd`'s
  (tested). Signs are fixed so each U column's largest entry is positive.

## Gotchas

- `imgui.push_font(None, size)` takes a size before `style.font_scale_main`; `widgets.push_size` divides it out.
- Text drawn at an explicit size is measured at that size (`diagram.text_width` pushes the font): widths measured at
  the base size and scaled come out short.
- `ImDrawList.add_rect` is (p_min, p_max, col, rounding, thickness, flags) and `add_polyline` is (points, col,
  thickness, flags): thickness before flags, unlike imgui's C++ order.
- Icons are FontAwesome 6 (`default_icon_font` in `runner_params`); hello_imgui's default is 4.
- `implot.add_colormap` asserts on a duplicate name, so `plotting.add_colormap` looks the name up first.
  Colormaps are made on a stage's first `page()`, because they need an ImPlot context.
- Headless tests run on hello_imgui's null backend, which needs `renderer_has_textures` set in `post_init`.
- `DebugTools.debugger` is a method, not a property.
