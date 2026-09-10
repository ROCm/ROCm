import { domReady, logDebug } from "./utils.js";
import {
  updateTOC2ContentsList,
  updateTOC2OptionsList,
} from "./selector-toc2.js";

const GROUP_QUERY = ".rocm-docs-selector-group";
const OPTION_QUERY = ".rocm-docs-selector-option";
const DROPDOWN_INPUT_QUERY = ".rocm-docs-selector-dropdown-input";
const COND_QUERY = "[data-show-cond],[data-disable-cond]";

const DEFAULT_OPTION_CLASS = "rocm-docs-selector-option-default";
const DISABLED_CLASS = "rocm-docs-disabled";
const HIDDEN_CLASS = "rocm-docs-hidden";
const SELECTED_CLASS = "rocm-docs-selected";

const STORAGE_KEY = "rocm-docs-selector-state";
// Guard against infinite reconciliation loops from circular state dependencies.
const MAX_VISIBILITY_CYCLES = 5;

// Toggle helpers -------------------------------------------------------------

const isDefaultOption = (elem) => elem.classList.contains(DEFAULT_OPTION_CLASS);

const disable = (elem) => {
  elem.classList.add(DISABLED_CLASS);
  elem.setAttribute("aria-disabled", "true");
  elem.setAttribute("tabindex", "-1");
};

const enable = (elem) => {
  elem.classList.remove(DISABLED_CLASS);
  elem.setAttribute("aria-disabled", "false");
  elem.setAttribute("tabindex", "0");
};

const hide = (elem) => {
  elem.classList.add(HIDDEN_CLASS);
  elem.setAttribute("aria-hidden", "true");
};

const show = (elem) => {
  elem.classList.remove(HIDDEN_CLASS);
  elem.setAttribute("aria-hidden", "false");
};

const select = (elem) => {
  elem.classList.add(SELECTED_CLASS);
  elem.setAttribute("aria-checked", "true");
};

const deselect = (elem) => {
  elem.classList.remove(SELECTED_CLASS);
  elem.setAttribute("aria-checked", "false");
};

// URL synchronization --------------------------------------------------------

function syncStateToURL() {
  const query = new URLSearchParams(state).toString();
  const newURL = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
  window.history.replaceState({}, "", newURL);
  logDebug("URL updated:", newURL);
}

function getStateFromURL() {
  return Object.fromEntries(new URLSearchParams(window.location.search));
}

// localStorage synchronization -----------------------------------------------

function syncStateToLocalStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    logDebug("localStorage updated:", state);
  } catch (err) {
    console.warn("[ROCmDocsSelector] Failed to save to localStorage:", err);
  }
}

function getStateFromLocalStorage() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return {};
    const parsed = JSON.parse(stored);
    logDebug("localStorage loaded:", parsed);
    return parsed;
  } catch (err) {
    console.warn("[ROCmDocsSelector] Failed to read from localStorage:", err);
    return {};
  }
}

// Global selector state ------------------------------------------------------

const state = {};

// Keys introduced exclusively via extra bindings (not by any selector group's
// primary key). Tracked so stale bindings can be cleared when the option that
// produced them is no longer selected.
const extraBindingKeys = new Set();

// Keys owned by at least one selector group as a primary key. Extra-binding
// cleanup must never delete these — reconcileGroupSelections manages them.
const primarySelectorKeys = new Set();

// True while updateVisibility is running; suppresses per-mutation URL/storage
// syncs so exactly one sync happens per user interaction.
let isBatchingState = false;

function syncState() {
  syncStateToURL();
  syncStateToLocalStorage();
}

function setState(updates) {
  Object.assign(state, updates);
  logDebug("State updated:", state);
  if (!isBatchingState) syncState();
}

function deleteStateKeys(keys) {
  for (const key of keys) {
    delete state[key];
  }
  if (!isBatchingState) syncState();
}

// Condition handling ---------------------------------------------------------

/**
 * Safely parse JSON-encoded conditions from a data-* attribute.
 * Expects a key/value object, where values may be strings or arrays of strings.
 */
function parseConditions(attrName, raw) {
  if (!raw) return null;

  try {
    const conditions = JSON.parse(raw);
    if (typeof conditions !== "object" || Array.isArray(conditions)) {
      console.warn(
        `[ROCmDocsSelector] Invalid '${attrName}' format ` +
          "(must be a key/value object):",
        raw,
      );
      return null;
    }
    return conditions;
  } catch (err) {
    console.error(
      `[ROCmDocsSelector] Couldn't parse '${attrName}' conditions:`,
      err,
    );
    return null;
  }
}

/**
 * Return true iff all conditions match the current state.
 * - Values can be a string or an array of strings.
 * - A condition with an undefined state key is treated as not matching.
 */
function matchesConditions(conditions, currentState) {
  for (const [key, expected] of Object.entries(conditions)) {
    const actual = currentState[key];
    if (actual === undefined) return false;
    if (Array.isArray(expected)) {
      if (!expected.includes(actual)) return false;
    } else if (actual !== expected) {
      return false;
    }
  }
  return true;
}

function shouldBeDisabled(elem) {
  const raw = elem.dataset.disableCond;
  if (!raw) return false;

  const conditions = parseConditions("disable-cond", raw);
  if (!conditions) {
    console.warn(
      "[ROCmDocsSelector] Invalid 'disable-cond' conditions; not disabling affected element.",
    );
    return false;
  }
  return matchesConditions(conditions, state);
}

function shouldBeShown(elem) {
  const raw = elem.dataset.showCond;
  if (!raw) return true;

  const conditions = parseConditions("show-cond", raw);
  if (!conditions) return true;
  return matchesConditions(conditions, state);
}

// Event handlers -------------------------------------------------------------

/**
 * Parse the data-selector-extra-bindings attribute into a plain object.
 * Returns {} if absent or invalid.
 */
function getExtraBindings(option) {
  const raw = option.dataset.selectorExtraBindings;
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * Select the option matching `value` for `key`, deselect all others.
 * Updates ARIA/class state on every matching OPTION_QUERY element so that
 * reconciliation, extra-binding logic, and TomSelect sync all see the
 * correct selection.
 */
function applySelectionByKey(key, value) {
  document
    .querySelectorAll(`${OPTION_QUERY}[data-selector-key="${key}"]`)
    .forEach((opt) => {
      if (opt.dataset.selectorValue === value) {
        select(opt);
      } else {
        deselect(opt);
      }
    });
}

/**
 * Recompute extra bindings from all currently-selected options and sync state.
 *
 * Called inside the updateVisibility loop so that reconcileGroupSelections
 * picking a new option (e.g. after a higher-level selector changes) also
 * triggers a fresh evaluation of extra bindings.
 *
 * Returns true if state changed.
 */
function reapplyAllExtraBindings() {
  // Collect desired extra bindings from every currently-selected option that
  // is visible (not hidden, not inside a hidden ancestor). Hidden groups are
  // skipped by reconcileGroupSelections, so their options may still carry
  // SELECTED_CLASS; including them would let a stale binding overwrite the
  // active one when multiple groups share the same key (e.g. "gpu").
  const desired = {};
  document.querySelectorAll(`${OPTION_QUERY}.${SELECTED_CLASS}`).forEach((opt) => {
    if (!opt.classList.contains(HIDDEN_CLASS) && !opt.closest(`.${HIDDEN_CLASS}`)) {
      Object.assign(desired, getExtraBindings(opt));
    }
  });

  // Never delete a key that belongs to a primary selector group — those are
  // managed by reconcileGroupSelections, not by extra-binding cleanup. Without
  // this guard, switching families (e.g. Instinct → Radeon) would delete "w"
  // from state because the Instinct option sets w=compute as an extra binding
  // but the Radeon option does not, even though a visible Use case selector
  // group owns "w" as its primary key.
  const staleKeys = Array.from(extraBindingKeys).filter(
    (k) => !(k in desired) && !primarySelectorKeys.has(k),
  );

  const updates = {};
  for (const [k, v] of Object.entries(desired)) {
    if (state[k] !== v) updates[k] = v;
  }

  // Rebuild tracking set to match current selections, excluding primary keys.
  extraBindingKeys.clear();
  for (const key of Object.keys(desired)) {
    if (!primarySelectorKeys.has(key)) extraBindingKeys.add(key);
  }

  let changed = false;
  if (staleKeys.length) {
    deleteStateKeys(staleKeys);
    changed = true;
  }
  if (Object.keys(updates).length) {
    // Sync the DOM before setState so reconcileGroupSelections doesn't
    // treat the stale DOM selection as authoritative on the next iteration.
    for (const [k, v] of Object.entries(updates)) {
      applySelectionByKey(k, v);
    }
    setState(updates);
    changed = true;
  }
  return changed;
}

function handleOptionSelect(e) {
  const option = e.currentTarget;

  // Ignore interaction with disabled or already selected options
  if (
    option.classList.contains(DISABLED_CLASS) ||
    option.classList.contains(SELECTED_CLASS)
  ) {
    return;
  }

  const { selectorKey: key, selectorValue: value } = option.dataset;
  if (!key || !value) return;

  applySelectionByKey(key, value);

  // Update global state. Extra bindings are applied by reapplyAllExtraBindings
  // inside the updateVisibility loop, so they are always derived from the
  // current selection rather than applied imperatively here.
  setState({ [key]: value });
  updateVisibility();
}

function handleOptionKeydown(e) {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    handleOptionSelect(e);
  }
}

// Visibility / enablement update ---------------------------------------------

// Ensure each selector group always has a valid selected option.
// If the current selection becomes disabled/hidden due to another selector's
// change, automatically pick a replacement.
function reconcileGroupSelections() {
  const currentState = { ...state };
  const updates = {};
  // Keys from hidden groups that may need to be cleared from state.
  const hiddenGroupKeys = [];
  // Keys owned by at least one visible group — used to protect against
  // stale-key deletion even when the selected value didn't change (i.e. the
  // key never entered `updates`).
  const claimedKeys = new Set();

  document.querySelectorAll(GROUP_QUERY).forEach((group) => {
    const options = Array.from(group.querySelectorAll(OPTION_QUERY));
    const groupKey =
      group.dataset.selectorKey || options[0]?.dataset.selectorKey;

    // Skip groups that are hidden OR inside a hidden parent.
    // Also remove the group's key from state so it doesn't persist in the URL.
    if (
      group.classList.contains(HIDDEN_CLASS) ||
      group.closest(`.${HIDDEN_CLASS}`)
    ) {
      if (groupKey && groupKey in state) hiddenGroupKeys.push(groupKey);
      return;
    }

    if (!options.length) return;
    if (!groupKey) return;

    claimedKeys.add(groupKey);

    // Options that are both enabled and visible
    const enabledVisible = options.filter(
      (opt) =>
        !opt.classList.contains(DISABLED_CLASS) &&
        !opt.classList.contains(HIDDEN_CLASS),
    );

    if (!enabledVisible.length) {
      options.forEach(deselect);
      return;
    }

    const currentlySelected = options.find((opt) =>
      opt.classList.contains(SELECTED_CLASS),
    );

    if (currentlySelected && enabledVisible.includes(currentlySelected)) {
      const selectedValue = currentlySelected.dataset.selectorValue;
      if (selectedValue && currentState[groupKey] !== selectedValue) {
        updates[groupKey] = selectedValue;
      }
      return;
    }

    // Prefer: persisted state value → marked default → first available
    const stateValue = currentState[groupKey];
    const replacement =
      (stateValue && enabledVisible.find((opt) => opt.dataset.selectorValue === stateValue)) ||
      enabledVisible.find(isDefaultOption) ||
      enabledVisible[0];

    if (!replacement) return;

    options.forEach(deselect);
    select(replacement);

    const newValue = replacement.dataset.selectorValue;
    if (newValue && currentState[groupKey] !== newValue) {
      updates[groupKey] = newValue;
    }
  });

  // Only delete a key if no visible group claimed it. Using claimedKeys (not
  // just `updates`) covers the case where the selection was already correct —
  // the key never enters `updates` but is still actively owned by a visible group.
  const staleKeys = hiddenGroupKeys.filter((k) => !claimedKeys.has(k));

  let changed = false;
  if (Object.keys(updates).length) {
    setState(updates);
    changed = true;
  }
  if (staleKeys.length) {
    deleteStateKeys(staleKeys);
    changed = true;
  }
  return changed;
}

// TomSelect instances for dropdown-input groups, keyed by selector key.
// Multiple groups can share the same key (e.g. "gpu" for Instinct/Radeon/Ryzen),
// so each key maps to an array of instances.
const dropdownInstances = new Map();

function syncDropdownsToState() {
  for (const [key, instances] of dropdownInstances) {
    const val = state[key];
    for (const ts of instances) {
      // Only sync the TomSelect whose parent group is currently visible.
      const groupElem = ts.input.closest(GROUP_QUERY);
      if (groupElem && (groupElem.classList.contains(HIDDEN_CLASS) || groupElem.closest(`.${HIDDEN_CLASS}`))) {
        continue;
      }
      if (val !== undefined && ts.getValue() !== val) {
        ts.setValue(val, true); // silent: suppress change event
      }
    }
  }
}

function flashNewlyVisible(condElems, hiddenBefore) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  // Collect elements that just became visible.
  const newlyVisible = Array.from(condElems).filter(
    (el) =>
      el.dataset.showCond !== undefined &&
      hiddenBefore.has(el) &&
      !el.classList.contains(HIDDEN_CLASS),
  );
  if (newlyVisible.length === 0) return;

  // Skip elements whose ancestor is also newly visible to avoid double-flashing.
  const newlyVisibleSet = new Set(newlyVisible);
  const toFlash = newlyVisible.filter((el) => {
    let node = el.parentElement;
    while (node) {
      if (newlyVisibleSet.has(node)) return false;
      node = node.parentElement;
    }
    return true;
  });

  // Batch all reads before any writes so the browser can compute layout once.
  const accentColor = getComputedStyle(document.documentElement)
    .getPropertyValue("--rocm-docs-selector-option-hover-color").trim();
  const prevColors = toFlash.map((el) => getComputedStyle(el).color);

  // Start all animations simultaneously — no per-element reflow needed.
  toFlash.forEach((el, i) => {
    el.animate(
      [{ color: accentColor }, { color: prevColors[i] }],
      { duration: 1000, easing: "ease-out", fill: "none" },
    );
  });
}

let isUpdatingVisibility = false;

function updateVisibility() {
  // Prevent re-entrancy if something triggers updateVisibility
  // while it is already running.
  if (isUpdatingVisibility) return;
  isUpdatingVisibility = true;
  isBatchingState = true;

  try {
    let stateChanged = false;
    let iterations = 0;

    // Hoist the conditional elements query outside the loop — the set of
    // elements with show/disable conditions doesn't change between iterations,
    // only their CSS classes and state do.
    const condElems = document.querySelectorAll(COND_QUERY);

    // Snapshot which show-cond elements are currently hidden before any changes.
    const hiddenBefore = new Set(
      Array.from(condElems).filter(
        (el) => el.dataset.showCond !== undefined && el.classList.contains(HIDDEN_CLASS),
      ),
    );

    // We may need multiple passes: reconciling selections can change the
    // global state, which in turn affects show/disable conditions.
    do {
      condElems.forEach((elem) => {
        if (elem.dataset.showCond !== undefined) {
          if (shouldBeShown(elem)) show(elem); else hide(elem);
        }
        if (elem.dataset.disableCond !== undefined) {
          if (shouldBeDisabled(elem)) disable(elem); else enable(elem);
        }
      });

      stateChanged = reconcileGroupSelections();
      // Re-evaluate extra bindings after every reconciliation pass so that a
      // newly-selected option's bindings take effect (and stale ones are cleared).
      stateChanged = reapplyAllExtraBindings() || stateChanged;
      iterations += 1;
    } while (stateChanged && iterations < MAX_VISIBILITY_CYCLES);

    // Single URL + localStorage sync after all state mutations are settled.
    syncState();

    flashNewlyVisible(condElems, hiddenBefore);
    updateTOC2OptionsList();
    updateTOC2ContentsList();
    syncDropdownsToState();
  } finally {
    isBatchingState = false;
    isUpdatingVisibility = false;
  }
}

// Initialization -------------------------------------------------------------

domReady(() => {
  const selectorOptions = document.querySelectorAll(OPTION_QUERY);
  if (!selectorOptions.length) {
    return;
  }

  const defaultState = {};
  const localStorageState = getStateFromLocalStorage();
  const urlState = getStateFromURL();

  // Collect primary selector keys and extra binding keys separately.
  // Used below to drop stale state and to pre-populate extraBindingKeys so
  // that stale URL/localStorage extra-binding values are cleared on init.
  const primaryPageKeys = new Set();
  const extraBindingPageKeys = new Set();

  selectorOptions.forEach((option) => {
    option.addEventListener("click", handleOptionSelect);
    option.addEventListener("keydown", handleOptionKeydown);

    const key = option.dataset.selectorKey;
    if (key) primaryPageKeys.add(key);
    for (const k of Object.keys(getExtraBindings(option))) extraBindingPageKeys.add(k);

    if (isDefaultOption(option)) {
      const value = option.dataset.selectorValue;
      if (key && value && !(key in defaultState)) {
        defaultState[key] = value;
        // Extra bindings for default options are applied by reapplyAllExtraBindings
        // inside the first updateVisibility() call below.
      }
    }
  });

  // Populate the module-level set so reapplyAllExtraBindings can guard against
  // deleting keys that are owned by primary selector groups.
  for (const key of primaryPageKeys) primarySelectorKeys.add(key);

  const pageKeys = new Set([...primaryPageKeys, ...extraBindingPageKeys]);

  // Merge with priority: URL > localStorage > defaults, then drop any key
  // that has no corresponding selector on this page so stale URL params and
  // localStorage entries from other pages are cleared immediately.
  const initialState = Object.fromEntries(
    Object.entries({ ...defaultState, ...localStorageState, ...urlState })
      .filter(([key]) => pageKeys.has(key)),
  );

  // Pre-populate extraBindingKeys with any state keys that are only known as
  // extra binding keys. This ensures reapplyAllExtraBindings() can identify
  // and clear them on the first updateVisibility() pass if no currently-
  // selected option requires them (e.g. gfx when fam=all is selected).
  for (const key of Object.keys(initialState)) {
    if (extraBindingPageKeys.has(key) && !primaryPageKeys.has(key)) {
      extraBindingKeys.add(key);
    }
  }

  // Initialize TomSelect dropdowns after initialState is known so each instance
  // can be immediately synced to the persisted state (URL / localStorage /
  // default) without waiting for the updateVisibility() pass.
  document.querySelectorAll(DROPDOWN_INPUT_QUERY).forEach((elem) => {
    const key = elem.dataset.selectorKey;
    if (!key) return;

    const ts = new TomSelect(elem);

    const initVal = initialState[key];
    if (initVal !== undefined && ts.getValue() !== initVal) {
      ts.setValue(initVal, true); // silent: suppress change event
    }

    if (!dropdownInstances.has(key)) dropdownInstances.set(key, []);
    dropdownInstances.get(key).push(ts);

    ts.on("change", (val) => {
      if (!val) return;
      applySelectionByKey(key, val);
      setState({ [key]: val });
      updateVisibility();
    });
  });

  // When the browser window loses focus, "blur" any focused TomSelect control.
  // Prevents dropdown flicker when the window is refocused.
  window.addEventListener("blur", () => {
    for (const instances of dropdownInstances.values()) {
      for (const ts of instances) {
        ts.blur();
      }
    }
  });

  for (const [key, value] of Object.entries(initialState)) {
    applySelectionByKey(key, value);
  }

  setState(initialState);
  updateVisibility();

  document.querySelectorAll(GROUP_QUERY).forEach((group) => {
    group.classList.add("rocm-docs-selector-initialized");
  });
});
